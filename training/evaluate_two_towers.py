import torch
import datasets
import faiss

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.two_towers import TwoTowers
from utils.tokeniser import Tokeniser

FAISS_INDEX_PATH = Path("weights/faiss_index.idx")
DOC_PATH = Path("dataset/internet/all_docs")

line_to_doc: dict[int, str] = {}
doc_to_line: dict[str, int] = {}
with open(DOC_PATH, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        line_to_doc[index] = line
        doc_to_line[line] = index

faiss_index: faiss.IndexFlatL2 = faiss.read_index(FAISS_INDEX_PATH)


hg_dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")["validation"]
tokeniser = Tokeniser()


def crude_eval(model: TwoTowers):
    model.eval()
    precisions = []
    with torch.inference_mode():
        for row in hg_dataset:
            query_string: str = row["query"]
            pos_doc_strings: list[str] = row["passages"]["passage_text"]

            pos_doc_ids: list[int] = [doc_to_line[doc] for doc in pos_doc_strings]

            query_tkns = tokeniser.tokenise_string(query_string)
            query_encoding = model.encode_query_single(query_tkns)

            query_vector = query_encoding.unsqueeze(0).cpu().numpy()

            _, pred_doc_ids = faiss_index.search(query_vector, len(pos_doc_strings))

            true_positives = sum(
                pred_doc_id in pos_doc_ids for pred_doc_id in pred_doc_ids
            )

            precision = true_positives / len(pred_doc_ids)
            precisions.append(precision)
            # recall = true_positives / len(pos_doc_ids)

    return sum(precisions) / len(precisions)
