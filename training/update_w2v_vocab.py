from pathlib import Path
import pickle
import sys
import os
import torch

project_root = r"/home/askar/mlx/two_towers/"

if project_root not in sys.path:
    sys.path.append(project_root)

print("Project root:", project_root)
print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from model.w2v_model import Word2Vec, EMBED_DIM
import wandb

old_lookup_path = Path("utils/lookup.pkl")
new_lookup_path = Path("utils/lookup_updated.pkl")

with open(old_lookup_path, "rb") as f:
    old_word_to_id, _ = pickle.load(f)

with open(new_lookup_path, "rb") as f:
    new_word_to_id, _ = pickle.load(f)

old_vocab_size = len(old_word_to_id)
new_vocab_size = len(new_word_to_id)

wandb.init(project="word2vec")
old_artifact = wandb.use_artifact("word2vec/word2vec_text8:latest", type="model")
artifact_dir = Path(old_artifact.download())

old_model = Word2Vec(EMBED_DIM, old_vocab_size)
old_model.load_state_dict(
    torch.load(artifact_dir / "w2v_epoch_80.pth", weights_only=True)
)
wandb.finish()

wandb.init(project="word2vec-marco")
new_model = Word2Vec(EMBED_DIM, new_vocab_size)

with torch.no_grad():
    new_model.center_embed.weight[:old_vocab_size] = old_model.center_embed.weight

new_save_dir = Path("checkpoints")

new_save_dir.mkdir(parents=True, exist_ok=True)

new_save_path = new_save_dir / "model.pth"
torch.save(new_model.state_dict(), new_save_path)
new_artifact = wandb.Artifact(
    "w2v-marco",
    type="model",
    metadata={"epoch": 0, "vocab_size": new_vocab_size, "embed_dim": EMBED_DIM},
)
new_artifact.add_file(new_save_path)
wandb.log_artifact(new_artifact)
wandb.finish()
