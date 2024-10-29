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

BATCH_SIZE = 5_000

two_tower_project = "search-towers"
wandb.init(project=two_tower_project)

print("Pulling model")
start_checkpoint_artifact = wandb.use_artifact(
    "askarkg12-personal/two-towers-marco/two-tower-model:v23", type="model"
)
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
train_dataset_path = Path("dataset/two_tower/train")
with open(train_dataset_path, "rb") as f:
    train_data = pickle.load(f)

val_dataset_path = Path("dataset/two_tower/train")
with open(val_dataset_path, "rb") as f:
    val_data = pickle.load(f)
print("Loaded dataset")

# dataset = TwoTowerDataset(data)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
# )
print("Dataset loader ready")

num_batches = math.ceil(len(train_data) / BATCH_SIZE)

for epoch in range(start_epoch + 1, start_epoch + 501):
    # Training
    batches = more_itertools.chunked(train_data, BATCH_SIZE)
    prgs = tqdm(batches, desc=f"Epoch {epoch}", total=num_batches)
    model.train()
    train_losses = []
    for batch in prgs:
        queries, pos, negs = zip(*batch)
        loss: torch.Tensor = model.get_loss_batch(queries, pos, negs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item()})
        train_losses.append(loss.item())
    wandb.log({"train_loss_epoch": sum(train_losses) / len(train_losses)})

    # Tracking validatoin loss
    model.eval()
    val_batches = more_itertools.chunked(val_data, BATCH_SIZE)
    val_losses = []

    pos_distances = []
    neg_distances = []
    with torch.inference_mode():
        for batch in val_batches:
            queries, pos, negs = zip(*batch)
            loss, pos_distance, neg_distance = model.get_loss_batch_with_distances(
                queries, pos, negs
            )
            pos_distances.append(pos_distance)
            neg_distances.append(neg_distance)
            val_losses.append(loss.item())
    wandb.log({"val_loss_epoch": sum(val_losses) / len(val_losses)})
    wandb.log({"val_pos_distance_epoch": sum(pos_distances) / len(pos_distances)})
    wandb.log({"val_neg_distance_epoch": sum(neg_distances) / len(neg_distances)})

    # Track accuracy cheap
    if not epoch % 10:
        # For each batch of query, pos and neg
        # Take cos_sim between query and pos
        # If cos_sim > threshold, count as correct
        pass

    # Tracking accuracy expensive
    if not epoch % 10:
        # Recache all doc encodings

        pass
    # Save checkpoint
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
