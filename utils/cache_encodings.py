from pathlib import Path
import math
import torch
from more_itertools import chunked
from tqdm import tqdm

import sys

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.two_towers import TwoTowers
from utils.tokeniser import Tokeniser

DOCS_PATH = Path("dataset/internet/all_docs")

USE_WANDB = True
BATCH_SIZE = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

if USE_WANDB:
    import wandb

    two_tower_project = "two-towers-marco"
    wandb.init(project=two_tower_project)

    print("Pulling model")
    start_checkpoint_artifact = wandb.use_artifact(
        "two-tower-model:latest", type="model"
    )
    artifact_dir = Path(start_checkpoint_artifact.download())
    start_epoch = start_checkpoint_artifact.metadata["epoch"]
    vocab_size = start_checkpoint_artifact.metadata["vocab_size"]
    encoding_dim = start_checkpoint_artifact.metadata["encoding_dim"]
    embed_dim = start_checkpoint_artifact.metadata["embedding_dim"]

    model_path = artifact_dir / "model.pth"
else:
    model_path = Path("app/weights/tt_model.pth")
    from app.processor import vocab_size, embed_dim, encoding_dim


model = TwoTowers(
    vocab_size=vocab_size, token_embed_dims=embed_dim, encoded_dim=encoding_dim
).to(device)

model.load_state_dict(
    torch.load(model_path, weights_only=True, map_location=map_location)
)
print("Model loaded")

tokeniser = Tokeniser()

doc_encodings_list = []

with open(DOCS_PATH, "rb") as f:
    total = sum(1 for _ in f)

early_stop_count = 0

with open(DOCS_PATH, "r", encoding="utf-8") as f:

    for chunk in tqdm(
        chunked(f, BATCH_SIZE),
        desc="Encoding documents",
        total=math.ceil(total / BATCH_SIZE),
    ):
        doc_tokens = [tokeniser.tokenise_string(doc) for doc in chunk]

        doc_encodings = model.encode_docs(doc_tokens)

        doc_encodings_list.append(doc_encodings)

        early_stop_count += 1
        if early_stop_count > 10:
            break

doc_encodings = torch.cat(doc_encodings_list, dim=0)

torch.save(doc_encodings, "app/weights/doc_encodings.pth")
