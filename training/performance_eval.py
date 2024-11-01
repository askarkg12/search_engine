from pathlib import Path
import sys
import torch
import torch.nn as nn
import datasets
import faiss
from tqdm import tqdm
import numpy as np

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.tokeniser import Tokeniser
from model.two_towers_modular import TwoTowers

dataset = datasets.load_dataset("imdb")

ALL_DOCS_PATH = root_dir / "dataset" / "two_tower" / "all_docs.txt"
BATCH_SIZE = 1024

RANK_CUTOFF = 100

# Could have done it faster, by counting lines when building the index
# but this is is easier to read
with open(ALL_DOCS_PATH, "rb") as f:
    total_docs_count = sum(1 for _ in f)

index_to_doc: dict[int, str] = {}
doc_to_index: dict[str, int] = {}
with open(ALL_DOCS_PATH, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        line = line.rstrip()
        index_to_doc[index] = line
        doc_to_index[line] = index


def evaluate_performance_two_towers(
    model: TwoTowers,
    tokeniser: Tokeniser,
    dataset_split,
    faiss_index: faiss.IndexFlatIP,
) -> float:

    keys_of_interest = ["query", "passages"]
    zipped_dataset_split = [
        dict(zip(keys_of_interest, items))
        for items in zip(*(dataset_split[key] for key in keys_of_interest))
    ]
    all_scores = []
    for row in tqdm(zipped_dataset_split, desc="Evaluating performance"):
        query = row["query"]

        pos_docs: list[str] = row["passages"]["passage_text"]
        pos_docs = [doc.rstrip() for doc in pos_docs]
        pos_doc_indices = [doc_to_index[doc] for doc in pos_docs]

        query_tokens = torch.tensor(tokeniser.tokenise_string(query))
        query_encoding_tensor: torch.Tensor = model(query_tokens).detach().cpu()
        query_encoding = query_encoding_tensor.unsqueeze(0).numpy()
        query_encoding_normalized = query_encoding / np.linalg.norm(
            query_encoding, axis=1, keepdims=True
        )
        _, pred_indices = faiss_index.search(query_encoding_normalized, RANK_CUTOFF)

        query_scores = []
        for pos_index in pos_doc_indices:
            if pos_index in pred_indices:
                rank = np.where(pred_indices == pos_index)[0][0]
                score = 1 - rank / RANK_CUTOFF
            else:
                score = 0
            query_scores.append(score)
        all_scores.append(np.mean(query_scores))
    return np.mean(all_scores)
