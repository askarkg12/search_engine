import pickle
from pathlib import Path
import sys
import torch
import more_itertools
import torch.optim as optim
import math
import wandb
from tqdm import tqdm

# This is not a very elegant solution, but it works for now
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
from model.two_towers import TwoTowers
from utils.two_tower_dataset import TwoTowerDataset, collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

BATCH_SIZE = 10_000

two_tower_project = "two-towers-marco"
wandb.init(project=two_tower_project)

print("Pulling model")
start_checkpoint_artifact = wandb.use_artifact("two-tower-model:latest", type="model")
artifact_dir = Path(start_checkpoint_artifact.download())
start_epoch = start_checkpoint_artifact.metadata["epoch"]
vocab_size = start_checkpoint_artifact.metadata["vocab_size"]
encoding_dim = start_checkpoint_artifact.metadata["encoding_dim"]
embed_dim = start_checkpoint_artifact.metadata["embedding_dim"]

model = TwoTowers(
    vocab_size=vocab_size, token_embed_dims=embed_dim, encoded_dim=encoding_dim
).to(device)

model.load_state_dict(
    torch.load(artifact_dir / "model.pth", weights_only=True, map_location=map_location)
)
print("Model pulled")

optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Loading dataset")
dataset_path = Path("dataset/two_tower/train")
with open(dataset_path, "rb") as f:
    data = pickle.load(f)
print("Loaded dataset")

# dataset = TwoTowerDataset(data)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
# )
print("Dataset loader ready")

num_batches = math.ceil(len(data) / BATCH_SIZE)

for epoch in range(start_epoch + 1, start_epoch + 501):
    batches = more_itertools.chunked(data, BATCH_SIZE)
    prgs = tqdm(batches, desc=f"Epoch {epoch}", total=num_batches)
    for batch in prgs:
        queries, pos, negs = zip(*batch)
        loss: torch.Tensor = model.get_loss_batch(queries, pos, negs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
    if not (epoch + 1) % 5:
        checkpoint_path = Path("artifacts/two-tower-model/model.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        new_artifact = wandb.Artifact(
            "two-tower-model",
            type="model",
            metadata={
                "epoch": epoch,
                "vocab_size": vocab_size,
                "encoding_dim": encoding_dim,
                "embedding_dim": embed_dim,
            },
        )
        new_artifact.add_file(checkpoint_path)
        wandb.log_artifact(new_artifact)
wandb.finish()
