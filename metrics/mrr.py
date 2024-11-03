from pathlib import Path
import sys

import faiss
import torch
import numpy as np
from typing import TypeAlias

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.tt_models import GenericEncoder

PaddedSequencesTensor: TypeAlias = torch.Tensor
LengthsTensor: TypeAlias = torch.Tensor
PosDocIds: TypeAlias = list[int]


def lazy_mrr(
    faiss_index: faiss.IndexFlatL2,
    batch: tuple[PaddedSequencesTensor, LengthsTensor, list[list[int]]],
    model: GenericEncoder,
    rank_cutoff: int = 100,
) -> float:
    # For this batch, get query encodings
    padded_queries, query_lens, pos_doc_ids = batch

    # Shape: (batch_size, hidden_dim)
    encoded_queries: torch.Tensor = model(padded_queries, query_lens)

    encodings_numpy = encoded_queries.detach().cpu().numpy()

    # Shape: (batch_size, hidden_dim)
    normalised_encodings = (encodings_numpy) / np.linalg.norm(
        encodings_numpy, axis=1, keepdims=True
    )

    _, pred_indices = faiss_index.search(normalised_encodings, rank_cutoff)

    mrr_scores = []
    for query_idx, pos_doc_id in enumerate(pos_doc_ids):
        if pos_doc_id in pred_indices[query_idx]:
            # If none of the predicted docs match the positive doc, MRR score is 0
            # Otherwise, use reciprocal of rank (1-based)
            rank = np.where(pred_indices[query_idx] == pos_doc_id)[0]
            if len(rank) > 0:
                mrr_scores.append(1 / (rank[0] + 1))
            else:
                mrr_scores.append(0)
    return np.mean(mrr_scores).item()
