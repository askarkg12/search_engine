import torch
import faiss
import numpy as np
from pathlib import Path
import sys

root_repo = Path(__file__).parent.parent
sys.path.append(str(root_repo))


from model.two_towers_modular import TwoTowers
from utils.tokeniser import Tokeniser
from pathlib import Path
from torch.nn.functional import cosine_similarity

# CACHED_ENCODINGS_PATH = Path("weights/doc_encodings.pth")
FAISS_INDEX_PATH = root_repo / "weights/server/faiss_index.idx"
TT_MODEL_PATH = root_repo / "weights/server/tt_weights.pth"
DOCS_PATH = root_repo / "dataset/two_tower/all_docs.txt"

USE_GENSIM = False

# TODO: Get these from the model metadata or config
VOCAB_SIZE = 81_547
ENCODING_DIM = 400
EMBED_DIM = 300 if USE_GENSIM else 50

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

print("Loading model")
if not USE_GENSIM:
    model = TwoTowers(
        vocab_size=VOCAB_SIZE, token_embed_dims=EMBED_DIM, encoded_dim=ENCODING_DIM
    ).to(device)
else:
    model = TwoTowers(
        vocab_size=VOCAB_SIZE,
        token_embed_dims=EMBED_DIM,
        encoded_dim=ENCODING_DIM,
        use_gensim=True,
    ).to(device)

model.load_state_dict(
    torch.load(TT_MODEL_PATH, weights_only=True, map_location=map_location),
    strict=False,
)
print("Model loaded")

tokeniser = Tokeniser()


faiss_index: faiss.IndexFlatIP = faiss.read_index(str(FAISS_INDEX_PATH))


# index_to_doc: list[str] = []
# with open(DOCS_PATH, "r", encoding="utf-8") as f:
#     for index, line in enumerate(f):
#         line = line.rstrip()
#         index_to_doc.append(line)
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    index_to_doc = [line.rstrip() for line in f]


def process_query(query: str):
    tokens = torch.tensor(tokeniser.tokenise_string(query), dtype=torch.long)
    query_encoding = (
        model.encode_query_single(tokens).unsqueeze(0).detach().cpu().numpy()
    )

    query_encoding_normalized = query_encoding / np.linalg.norm(
        query_encoding, axis=1, keepdims=True
    )

    _, top_k_indices = faiss_index.search(query_encoding_normalized, 10)

    return [index_to_doc[i] for i in top_k_indices[0]]


if __name__ == "__main__":
    askar_resp = process_query("King queen")
    same_resp = process_query("Sam")
    pass
