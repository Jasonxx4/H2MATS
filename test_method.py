import torch
import numpy as np
from tqdm import tqdm


def test_spa_model(s_emb, label, device, batch_size=256, top_k=50):
    s_emb = s_emb.to(device)
    emb_norm = (s_emb * s_emb).sum(dim=1).view(1, -1)
    label_r = []
    k = min(top_k, max(s_emb.size(0) - 1, 1))

    for i in tqdm(range(0, s_emb.size(0), batch_size)):
        c_batch = s_emb[i:i + batch_size]
        batch_norm = (c_batch * c_batch).sum(dim=1).view(-1, 1)
        distances_batch = batch_norm + emb_norm - 2 * torch.matmul(c_batch, s_emb.t())
        distances_batch = torch.clamp(distances_batch, min=0.0)

        row_count = distances_batch.size(0)
        row_idx = torch.arange(row_count, device=device)
        col_idx = torch.arange(i, i + row_count, device=device)
        distances_batch[row_idx, col_idx] = float('inf')

        label_r.append(torch.topk(distances_batch, k=k, dim=1, largest=False).indices.cpu())

    label_r = torch.cat(label_r, dim=0).numpy()
    recall = torch.zeros((s_emb.shape[0], 6))
    label = label.cpu().numpy() if torch.is_tensor(label) else label
    for idx, la in tqdm(enumerate(label)):
        recall[idx, 0] += len(list(set(label_r[idx, :10]).intersection(set(la[:10]))))  # HR-10
        recall[idx, 1] += len(list(set(label_r[idx, :50]).intersection(set(la[:50]))))  # HR-50
        recall[idx, 2] += len(list(set(label_r[idx, :50]).intersection(set(la[:10]))))  # R10@50
        recall[idx, 3] += len(list(set(label_r[idx, :1]).intersection(set(la[:1]))))  # R1@1
        recall[idx, 4] += len(list(set(label_r[idx, :10]).intersection(set(la[:1]))))  # R1@10
        recall[idx, 5] += len(list(set(label_r[idx, :50]).intersection(set(la[:1]))))  # R1@50

    return recall
