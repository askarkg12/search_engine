import torch
import pickle
from model.two_towers import TwoTowers, ENCODING_DIM
from model.w2v_model import EMBED_DIM
from utils.tokeniser import Tokeniser
from pathlib import Path
from torch.nn.functional import cosine_similarity

CACHED_ENCODINGS_PATH = Path("app/weights/cached_encodings.pkl")
TT_MODEL_PATH = Path("app/weights/tt_model.pth")
DOCS_PATH = Path("dataset/internet/all_docs")

VOCAB_SIZE = 81_547

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

print("Loading model")
model = TwoTowers(
    vocab_size=VOCAB_SIZE, token_embed_dims=EMBED_DIM, encoded_dim=ENCODING_DIM
).to(device)

model.load_state_dict(
    torch.load(TT_MODEL_PATH, weights_only=True, map_location=map_location)
)
print("Model loaded")

tokeniser = Tokeniser()

with open(CACHED_ENCODINGS_PATH, "rb") as f:
    cached_encodings = pickle.load(f)

# Shape: (num_encodings, encoding_dim)
cached_encodings_tensors = torch.tensor(cached_encodings, device=device)


def get_top_k_matches(query_encoding: torch.Tensor, k: int = 10):
    with torch.inference_mode():
        query_encoding = model(query_encoding)

        all_sims = cosine_similarity(
            cached_encodings_tensors, query_encoding.unsqueeze(0), dim=1
        )

        _, sorted_indices = torch.topk(all_sims, k)

        return sorted_indices


def get_line_from_index(index: int):
    with open(DOCS_PATH, "r") as f:
        f.seek(index)
        return f.readline()


def process_query(query: str):
    tokens = tokeniser.tokenise_string(query)
    query_encoding = model.encode_query_single(tokens)

    top_k_indices = get_top_k_matches(query_encoding)

    return [get_line_from_index(i) for i in top_k_indices]
