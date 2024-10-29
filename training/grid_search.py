from pathlib import Path
import torch
import torch.optim as optim
import sys
import pickle
from tqdm import tqdm
import wandb
import more_itertools
import math

root_dir = Path(__file__).parent.parent

sys.path.append(str(root_dir))

from model.two_towers_modular import TwoTowers

BATCH_SIZE = 1024

W2V_EMBED_PATH = root_dir / "model/weights/w2v_embeddings.pth"

DATASET_FILEPATH = root_dir / "dataset/two_tower"

TRAIN_FILEPATH = DATASET_FILEPATH / "train_strings.pkl"
VALIDATION_FILEPATH = DATASET_FILEPATH / "validation_strings.pkl"

with open(TRAIN_FILEPATH, "rb") as f:
    train_data = pickle.load(f)

with open(VALIDATION_FILEPATH, "rb") as f:
    validation_data = pickle.load(f)

total_train_len = math.ceil(len(train_data) / BATCH_SIZE)
total_val_len = math.ceil(len(validation_data) / BATCH_SIZE)

EPOCHS = 10

MODEL_CONFIGS = [
    {
        "run_name": "no_gensim_50",
        "use_gensim": False,
        "encoded_dim": 50,
        "optimizer": "adam",
        "lr": 0.001,
    },
    {
        "run_name": "no_gensim_100",
        "use_gensim": False,
        "encoded_dim": 100,
        "optimizer": "adam",
        "lr": 0.001,
    },
    {
        "run_name": "no_gensim_200",
        "use_gensim": False,
        "encoded_dim": 200,
        "optimizer": "adam",
        "lr": 0.001,
    },
    {"run_name": "gensim", "use_gensim": True, "optimizer": "adam", "lr": 0.001},
]


for config in MODEL_CONFIGS:
    run_name = config["run_name"]
    use_gensim = config["use_gensim"]
    encoded_dim = config["encoded_dim"]
    optimizer = config["optimizer"]
    lr = config["lr"]

    model = TwoTowers(
        token_embed_dims=300 if use_gensim else 50,
        encoded_dim=encoded_dim,
        use_gensim=use_gensim,
    )

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.init(project="two_tower_grid_search", name=run_name, config=config)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_chunks = more_itertools.chunked(train_data, BATCH_SIZE)
        loss_list = []
        pos_dist_list = []
        neg_dist_list = []
        for batch in tqdm(
            train_chunks,
            desc=f"Training epoch {epoch}",
            total=total_train_len,
            unit_scale=BATCH_SIZE,
        ):
            loss: torch.Tensor
            pos_dist: torch.Tensor
            neg_dist: torch.Tensor
            loss, (pos_dist, neg_dist) = model.get_loss_batch(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
            pos_dist_list.append(pos_dist.item())
            neg_dist_list.append(neg_dist.item())

        wandb.log(
            {
                "train_loss": sum(loss_list) / len(loss_list),
                "train_pos_dist": sum(pos_dist_list) / len(pos_dist_list),
                "train_neg_dist": sum(neg_dist_list) / len(neg_dist_list),
            }
        )

        # Validation
        model.eval()
        validation_chunks = more_itertools.chunked(validation_data, BATCH_SIZE)
        val_loss_list = []
        val_pos_dist_list = []
        val_neg_dist_list = []
        with torch.inference_mode():
            for batch in tqdm(
                validation_chunks,
                desc=f"Validating epoch {epoch}",
                total=total_val_len,
                unit_scale=BATCH_SIZE,
            ):
                loss, (pos_dist, neg_dist) = model.get_loss_batch(batch)
                val_loss_list.append(loss.item())
                val_pos_dist_list.append(pos_dist.item())
                val_neg_dist_list.append(neg_dist.item())

        wandb.log(
            {
                "val_loss": sum(val_loss_list) / len(val_loss_list),
                "val_pos_dist": sum(val_pos_dist_list) / len(val_pos_dist_list),
                "val_neg_dist": sum(val_neg_dist_list) / len(val_neg_dist_list),
            }
        )
