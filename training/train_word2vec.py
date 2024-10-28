import pickle
from pathlib import Path
import math

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

import more_itertools
from model.w2v_model import Word2Vec, EMBED_DIM

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 500_000

wandb_project = "word2vec-marco"
wandb.init(project=wandb_project)
print("Pulling model")
start_checkpoint_artifact = wandb.use_artifact("w2v-marco:latest", type="model")
artifact_dir = Path(start_checkpoint_artifact.download())
start_epoch = start_checkpoint_artifact.metadata["epoch"]
vocab_size = start_checkpoint_artifact.metadata["vocab_size"]

model = Word2Vec(embedding_dim=EMBED_DIM, vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(artifact_dir / "model.pth", weights_only=True))

print("Model pulled")

optimizer = optim.Adam(model.parameters(), lr=0.005)


print("Loading dataset")
dataset_path = Path("dataset/ms_marco/w2v_finetune.pkl")
with open(dataset_path, "rb") as f:
    inputs, targets, negs = pickle.load(f)

# combined = list(zip(inputs, targets, negs))
print("Loaded dataset")

print("Tensorising dataset")
input_tensor = torch.tensor(data=inputs, device=device, dtype=torch.long)
target_tensor = torch.tensor(data=targets, device=device, dtype=torch.long)
negs_tensor = torch.tensor(data=negs, device=device, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor, negs_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Dataset loader ready")


for epoch in range(start_epoch + 1, start_epoch + 501):
    # for chunk in tqdm(more_itertools.chunked(combined, math.ceil(len(combined) / 5))):
    # input_tensor = torch.LongTensor(chunk[0]).to(device)
    # target_tensor = torch.LongTensor(chunk[1]).to(device)
    # negs_tensor = torch.LongTensor(chunk[2]).to(device)
    # # input_tensor, target_tensor, negs_tensor = torch.load(load_path)

    # dataset = torch.utils.data.TensorDataset(
    #     input_tensor, target_tensor, negs_tensor
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, shuffle=True
    # )
    prgs = tqdm(dataloader, desc=f"Epoch {epoch}")
    for inputs, targets, negs in prgs:
        inputs, targets, negs = (inputs, targets, negs)
        # inputs.to(device),
        # targets.to(device),
        # negs.to(device),
        optimizer.zero_grad()

        loss: torch.Tensor = model.get_loss(inputs, targets, negs)

        loss.backward()

        optimizer.step()
        wandb.log({"loss": loss.item()})
    if not (epoch + 1) % 5:
        # save_path = f"checkpoints/w2v_epoch_{epoch+1}.pth"
        checkpoint_dir = Path("artifacts/w2v-marco")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pth"

        artifact = wandb.Artifact
        torch.save(model.state_dict(), checkpoint_path)
        new_artifact = wandb.Artifact(
            "w2v-marco",
            type="model",
            metadata={"epoch": epoch, "vocab_size": vocab_size, "embed_dim": EMBED_DIM},
        )
        new_artifact.add_file(checkpoint_path)
        wandb.log_artifact(new_artifact)
wandb.finish()
