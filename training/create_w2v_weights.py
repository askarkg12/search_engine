import torch
import wandb
from pathlib import Path
import sys

# This is not a very elegant solution, but it works for now
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.w2v_model import Word2Vec, EMBED_DIM
from model.two_towers import TwoTowers, ENCODING_DIM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Pulling word2vec model")
vocab_size = 81_547

model = Word2Vec(embedding_dim=EMBED_DIM, vocab_size=vocab_size).to(device)
map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(
    torch.load(
        "weights/w2v_state_dict.pth", weights_only=True, map_location=map_location
    )
)
print("Model pulled")

w2v_embed_weights = model.center_embed.weight.data
embed_weights_path = Path("model/weights/w2v_embeddings.pth")
embed_weights_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(w2v_embed_weights, embed_weights_path)
