import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
from sequence_encoders import MambaSeqEncoder, BiMambaSeqEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device, dropout: float = 0.1, max_len: int = 3700):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.d_model = int(d_model)
        self.register_buffer('pe', self._build_encoding(max_len, torch.device('cpu'), torch.float32))

    def _build_encoding(self, length, device, dtype):
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, 1, self.d_model, device=device, dtype=dtype)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_length(self, length, device, dtype):
        if length <= self.pe.size(0):
            return
        # The sinusoidal encoding has no learned rows, so extrapolating it keeps
        # checkpoints trained with the original 3700-step buffer compatible.
        expanded = self._build_encoding(length, device, dtype)
        self.pe = expanded

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        self._ensure_length(x.size(0), x.device, x.dtype)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, device, activation_checkpointing=False):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, device, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            output = checkpoint(
                lambda checkpoint_src: self.transformer_encoder(
                    checkpoint_src,
                    src_key_padding_mask=src_mask,
                ),
                src,
                use_reentrant=False,
            )
        else:
            output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.decoder(output)
        return output


class PosKnnGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, usePE, useSI, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.usePE = usePE
        self.useSI = useSI
        logging.info('usePE: ' + str(self.usePE) + ', useSI: ' + str(self.useSI))
        self.dataset = dataset
        if dataset == 'beijing':
            self.nodeLin = torch.nn.Linear(in_channels + 98, in_channels, bias=False)
        elif dataset == 'tdrive':
            self.nodeLin = torch.nn.Linear(in_channels + 112, in_channels, bias=False)
        elif dataset == 'porto':
            self.nodeLin = torch.nn.Linear(in_channels + 68, in_channels, bias=False)

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.nodeLin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=[deg_norm, edge_norm])

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        if self.useSI:
            return norm[0].view(-1, 1) * (self.lin1(x_j)) + norm[1].view(-1, 1) * (self.lin2(x_j))
        else:
            return norm.view(-1, 1) * (self.lin1(x_j))

    def gcn_forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=deg_norm)

        return out


class KnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, useSI, dataset, alpha1, alpha2):
        super(KnnGNN, self).__init__()
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.posconv1 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv2 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv3 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv4 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data[0].x, data[0].edge_index, data[0].edge_attr
        _, edge_index_l1, edge_weight_l1 = data[1].x, data[1].edge_index, data[1].edge_attr
        _, edge_index_l2, edge_weight_l2 = data[2].x, data[2].edge_index, data[2].edge_attr

        x0 = F.relu(self.posconv1(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv2(x, edge_index_l1, edge_weight_l1, d2an, True))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1

        x0 = F.relu(self.posconv3(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv4(x, edge_index_l1, edge_weight_l1, d2an, False))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1
        return x

    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data.x, data.edge_index, data.edge_attr

        x0 = F.relu(self.posconv1.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        x0 = F.relu(self.posconv3.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        return x


class SMNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size,  stard_LSTM=False, incell=True, device=0,
                 seq_model='lstm', mamba_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super(SMNEncoder, self).__init__()
        self.input_size = input_size

        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        self.seq_model = seq_model
        self.mlp_ele = torch.nn.Linear(2, int(hidden_size/2))
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        self.nonTanh = torch.nn.Tanh()
        self.point_pooling = torch.nn.AvgPool1d(10)

        self.seq_model_layer = 1
        self.device = device
        if seq_model == 'lstm':
            self.t2s_model = torch.nn.LSTM(self.input_size, hidden_size, num_layers=self.seq_model_layer)
        elif seq_model == 'mamba':
            self.input_proj = torch.nn.Linear(self.input_size, hidden_size)
            self.t2s_model = MambaSeqEncoder(
                d_model=hidden_size,
                out_dim=hidden_size,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif seq_model == 'transformer':
            self.input_proj = torch.nn.Linear(self.input_size, hidden_size)
            self.t2s_model = TransformerModel(hidden_size, hidden_size, 4, 512, 1, 0.3, device)
        else:
            raise ValueError(f"Unknown coordinate sequence model: {seq_model}")

        self.res_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear3 = torch.nn.Linear(hidden_size, hidden_size)

    def obtain_src_mask(self, input_len, max_len):
        if not torch.is_tensor(input_len):
            input_len = torch.tensor(input_len, device=self.device)
        input_len = input_len.to(self.device)
        mask = torch.ones((input_len.size(0), max_len), device=self.device)
        for i, length in enumerate(input_len):
            length = int(length.item())
            if length <= max_len:
                mask[i, :length] = 0
        return mask.bool()

    def forward(self, inputs_a):
        input_a, input_len_a = inputs_a

        if self.seq_model == 'lstm':
            outputs_a, (hn_a, cn_a) = self.t2s_model(input_a.permute(1, 0, 2))
        elif self.seq_model == 'mamba':
            padding_mask = self.obtain_src_mask(input_len_a, input_a.size(1))
            outputs_a = self.t2s_model(self.input_proj(input_a), padding_mask=padding_mask).permute(1, 0, 2)
        else:
            padding_mask = self.obtain_src_mask(input_len_a, input_a.size(1))
            outputs_a = self.t2s_model(self.input_proj(input_a).permute(1, 0, 2), padding_mask)
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * F.tanh(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            v = int(v.item()) if torch.is_tensor(v) else int(v)
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)
        return fa_outputs, outputs_fa


class GraphTrajSimEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, hidden_size, num_layers, dropout_rate, concat, device,
                 usePE, useSI, useLSTM, dataset, alpha1, alpha2,
                 seq_encoder='transformer', mamba_layers=2, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 coord_encoder='lstm'):
        super(GraphTrajSimEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.usePE = usePE
        self.useSI = useSI
        self.useLSTM = useLSTM
        self.dataset = dataset
        self.seq_encoder_type = seq_encoder
        self.coord_encoder = coord_encoder
        self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE,
                                      self.useSI, self.dataset, alpha1, alpha2)
        if seq_encoder == 'transformer':
            self.trm_encoder = TransformerModel(hidden_size, embedding_size, 4, 512, 1, 0.3, device)
        elif seq_encoder == 'mamba':
            self.seq_encoder = MambaSeqEncoder(
                d_model=embedding_size,
                out_dim=hidden_size,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif seq_encoder == 'bimamba':
            self.seq_encoder = BiMambaSeqEncoder(
                d_model=embedding_size,
                out_dim=hidden_size,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        elif seq_encoder == 'hmt':
            raise ValueError("seqEncoder='hmt' requires useHybridGeo=True and hierarchy data.")
        else:
            raise ValueError(f"Unknown sequence encoder: {seq_encoder}")
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        logging.info('useLSTM: ' + str(self.useLSTM))
        if useLSTM:
            logging.info('coordEncoder: ' + str(self.coord_encoder))
            self.smn = SMNEncoder(2,
                                  hidden_size,
                                  stard_LSTM=True,
                                  incell=True,
                                  device=self.device,
                                  seq_model=coord_encoder,
                                  mamba_layers=mamba_layers,
                                  mamba_d_state=mamba_d_state,
                                  mamba_d_conv=mamba_d_conv,
                                  mamba_expand=mamba_expand).to(self.device)
            self.out_linear = nn.Linear(2*hidden_size, 2*hidden_size)

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l <= max_len:
                mask[i, :l] = 0

        return mask.bool()

    def forward(self, network_data, traj_seqs, coor_seqs, seq_lengths):
        seq_lengths = seq_lengths.to(self.device)
        traj_seqs = traj_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)
        # get node embeddings from gcn, (num_nodes, embedding_size)
        if self.useSI:
            graph_node_embeddings = self.graph_embedding(network_data)
        else:
            graph_node_embeddings = self.graph_embedding.noSI_forward(network_data)

        embedded_seq_tensor = graph_node_embeddings[traj_seqs].to(self.device)
        # (batch_size, max_len, embedding_size), trm
        trm_encoder_src_mask = self.obtain_trm_src_mask(seq_lengths)
        if self.seq_encoder_type == 'transformer':
            seq_outputs = self.trm_encoder(embedded_seq_tensor.transpose(1, 0), trm_encoder_src_mask).transpose(1, 0)
        else:
            seq_outputs = self.seq_encoder(embedded_seq_tensor, padding_mask=trm_encoder_src_mask)
        u = torch.tanh(torch.matmul(seq_outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(trm_encoder_src_mask == True, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = seq_outputs * att_score
        attrn_out = torch.sum(scored_outputs, dim=1)
        if self.useLSTM:
            anchor_embedding,  outputs_ap = self.smn([coor_seqs, seq_lengths])
            out = torch.cat((attrn_out, anchor_embedding), dim=-1)
            out = self.out_linear(out)
        else:
            out = attrn_out
        return out
