from pathlib import Path
import sys
import torch
import torch.nn as nn
import datasets
import faiss
from more_itertools import chunked
from tqdm import tqdm
import math
import numpy as np

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.tokeniser import Tokeniser
from model.two_towers_modular import TwoTowers

dataset = datasets.load_dataset("imdb")

ALL_DOCS_PATH = root_dir / "data" / "all_docs"
BATCH_SIZE = 100

RANK_CUTOFF = 100

# Could have done it faster, by counting lines when building the index
# but this is is easier to read
with open(ALL_DOCS_PATH, "rb") as f:
    total_docs_count = sum(1 for _ in f)

index_to_doc: dict[int, str] = {}
doc_to_index: dict[str, int] = {}
with open(ALL_DOCS_PATH, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        index_to_doc[index] = line
        doc_to_index[line] = index

all_doc_encodings_list: list[np.ndarray] = []


def build_doc_faiss_index(model: TwoTowers, tokeniser: Tokeniser) -> float:
    # Build doc faiss index
    with torch.inference_mode():
        with open(ALL_DOCS_PATH, "r") as f:
            for chunk in tqdm(
                chunked(f, BATCH_SIZE),
                desc="Encoding documents",
                total=math.ceil(total_docs_count / BATCH_SIZE),
            ):
                doc_tokens = [tokeniser.tokenise_string(doc) for doc in chunk]

                doc_encodings = model.encode_docs(doc_tokens).detach().numpy()

                all_doc_encodings_list.append(doc_encodings)

    all_doc_encodings = np.concatenate(all_doc_encodings_list, axis=0)
    index = faiss.IndexFlatIP(all_doc_encodings.shape[1])
    index.add(all_doc_encodings)
    return index


def evaluate_performance_two_towers(
    model: TwoTowers,
    tokeniser: Tokeniser,
    dataset_split,
    faiss_index: faiss.IndexFlatIP,
) -> float:

    for i, row in enumerate(dataset_split):
        query = row["text"]

        pos_docs = row["passages"]["passage_text"]
        pos_doc_indices = [doc_to_index[doc] for doc in pos_docs]

        query_tokens = tokeniser.tokenise_string(query)
        query_encoding = model(query_tokens)
        _, pred_indices = faiss_index.search(query_encoding, RANK_CUTOFF)

        all_scores = []
        for pos_index in pos_doc_indices:
            if pos_index in pred_indices:
                rank = np.where(pred_indices == pos_index)[0][0]
                score = 1 - rank / RANK_CUTOFF
                all_scores.append(score)

    return np.mean(all_scores)
