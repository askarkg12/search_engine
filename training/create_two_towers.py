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

word2vec_project = "word2vec-marco"
wandb.init(project=word2vec_project)
print("Pulling word2vec model")
start_checkpoint_artifact = wandb.use_artifact("w2v-marco:latest", type="model")
artifact_dir = Path(start_checkpoint_artifact.download())
start_epoch = start_checkpoint_artifact.metadata["epoch"]
vocab_size = start_checkpoint_artifact.metadata["vocab_size"]

model = Word2Vec(embedding_dim=EMBED_DIM, vocab_size=vocab_size).to(device)
map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(
    torch.load(artifact_dir / "model.pth", weights_only=True, map_location=map_location)
)
wandb.finish()
print("Model pulled")

two_tower = TwoTowers(
    vocab_size=vocab_size, token_embed_dims=EMBED_DIM, encoded_dim=ENCODING_DIM
)

with torch.no_grad():
    two_tower.embed.weight.data = model.center_embed.weight.data

two_tower_project = "two-towers-marco"
wandb.init(project=two_tower_project)
model_save_path = Path("model/checkpoints/two_tower_model/model.pth")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(two_tower.state_dict(), model_save_path)
artifact = wandb.Artifact(
    "two-tower-model",
    type="model",
    description="Two tower model for MS Marco",
    metadata={
        "epoch": 0,
        "vocab_size": vocab_size,
        "embedding_dim": EMBED_DIM,
        "encoding_dim": ENCODING_DIM,
    },
)
artifact.add_file(model_save_path)
wandb.log_artifact(artifact)
wandb.finish()
