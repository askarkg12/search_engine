from pathlib import Path
import faiss
import math
import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm

import sys

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.two_towers_modular import TwoTowers
from training.performance_eval import ALL_DOCS_PATH, BATCH_SIZE, total_docs_count
from utils.tokeniser import Tokeniser

DOCS_PATH = Path("dataset/internet/all_docs")


device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)


def build_doc_faiss_index(model: TwoTowers, tokeniser: Tokeniser) -> faiss.IndexFlatIP:
    all_doc_encodings_list: list[np.ndarray] = []
    # Build doc faiss index
    with torch.inference_mode():
        with open(ALL_DOCS_PATH, "r") as f:
            for chunk in tqdm(
                chunked(f, BATCH_SIZE),
                desc="Encoding documents",
                total=math.ceil(total_docs_count / BATCH_SIZE),
            ):
                doc_tokens = [
                    torch.tensor(tokeniser.tokenise_string(doc)) for doc in chunk
                ]

                doc_encodings = model.encode_docs(doc_tokens).detach().cpu().numpy()

                all_doc_encodings_list.append(doc_encodings)

    all_doc_encodings = np.concatenate(all_doc_encodings_list, axis=0)
    index = faiss.IndexFlatIP(all_doc_encodings.shape[1])
    all_doc_encodings_normalized = all_doc_encodings / np.linalg.norm(
        all_doc_encodings, axis=1, keepdims=True
    )
    index.add(all_doc_encodings_normalized)
    return index


if __name__ == "__main__":
    # Load model
    model = TwoTowers(
        token_embed_dims=50,
        encoded_dim=400,
        use_gensim=False,
    )
    checkpoint_path = Path("weights/server/tt_weights.pth")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=map_location, weights_only=True),
        strict=False,
    )
    tokeniser = Tokeniser(use_gensim=False)
    model.eval()
    with torch.inference_mode():
        faiss_index = build_doc_faiss_index(model, tokeniser)

    faiss.write_index(faiss_index, str(repo_root / "weights/server/faiss_index.idx"))
