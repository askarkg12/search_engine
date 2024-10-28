import torch
import faiss

from pathlib import Path
import sys

root_repo = Path(__file__).parent.parent
sys.path.append(str(root_repo))


from model.two_towers import TwoTowers, ENCODING_DIM
from model.w2v_model import EMBED_DIM
from utils.tokeniser import Tokeniser
from pathlib import Path
from torch.nn.functional import cosine_similarity

# CACHED_ENCODINGS_PATH = Path("weights/doc_encodings.pth")
FAISS_INDEX_PATH = Path("weights/faiss_index.faiss")
TT_MODEL_PATH = Path("weights/tt_weights.pth")
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


# Shape: (num_encodings, encoding_dim)
# cached_encodings_tensors = torch.tensor(cached_encodings, device=device)

# def get_top_k_matches(query_encoding: torch.Tensor, k: int = 10):
#     with torch.inference_mode():
#         query_encoding = model(query_encoding)

#         all_sims = cosine_similarity(
#             cached_encodings_tensors, query_encoding.unsqueeze(0), dim=1
#         )

#         _, sorted_indices = torch.topk(all_sims, k)

#         return sorted_indices

faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))


def get_line_from_index(index: int):
    with open(DOCS_PATH, "r",encoding="utf-8") as f:
        for doc_index, line in enumerate(f):
            if index == doc_index:
                return line
        return "LINE NOT FOUND"
        #f.seek(index)
        #return f.readline()


def process_query(query: str):
    tokens = tokeniser.tokenise_string(query)
    query_encoding = model.encode_query_single(tokens).unsqueeze(0).detach().cpu().numpy()

    _, top_k_indices = faiss_index.search(query_encoding, 10)

    return [get_line_from_index(i) for i in top_k_indices[0]]

if __name__=='__main__':
    for i in range(5):
        print(get_line_from_index(i))
    askar_resp = process_query('King queen')
    same_resp = process_query('Sam')
    pass