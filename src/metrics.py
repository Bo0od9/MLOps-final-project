from typing import Dict, List

import torch


def compute_hitrate(all_scores: torch.Tensor, positive_scores: torch.Tensor, k: int) -> List[float]:
    """
    Computes Hit Rate@k for each sample in the batch.
    """
    pos = positive_scores.unsqueeze(-1)
    cnt = (all_scores > pos).sum(dim=1)
    hits = cnt < k

    return hits.tolist()


def compute_dcg(all_scores: torch.Tensor, positive_scores: torch.Tensor, k: int) -> List[float]:
    """
    Computes DCG@k for each sample in the batch.
    """
    pos = positive_scores.unsqueeze(-1)
    cnt = (all_scores > pos).sum(dim=1)
    dcg = torch.zeros_like(cnt, dtype=torch.float32)
    hits = cnt < k
    ranks = cnt[hits]
    dcg[hits] = 1.0 / torch.log2(ranks + 2.0)

    return dcg.tolist()


def compute_metrics(all_scores: torch.Tensor, positive_scores: torch.Tensor) -> Dict[str, float]:
    return {
        "dcg@10": compute_dcg(all_scores, positive_scores, k=10),
        "dcg@100": compute_dcg(all_scores, positive_scores, k=100),
        "dcg@1000": compute_dcg(all_scores, positive_scores, k=1000),
        "hitrate@10": compute_hitrate(all_scores, positive_scores, k=10),
        "hitrate@100": compute_hitrate(all_scores, positive_scores, k=100),
        "hitrate@1000": compute_hitrate(all_scores, positive_scores, k=1000),
    }
