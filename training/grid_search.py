from pathlib import Path
import datasets
import torch
import torch.optim as optim
import sys
import pickle
from tqdm import tqdm
import wandb
import more_itertools
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.two_towers_modular import TwoTowers
from utils.tokeniser import Tokeniser
from training.performance_eval import (
    evaluate_performance_two_towers,
    build_doc_faiss_index,
)

BATCH_SIZE = 1024

PERFORMANCE_EVAL_EVERY_N_EPOCHS = 10


W2V_EMBED_PATH = root_dir / "model/weights/w2v_embeddings.pth"

DATASET_FILEPATH = root_dir / "dataset/two_tower"

GENSIM_TRAIN_FILEPATH = DATASET_FILEPATH / "train_gensim.pkl"
GENSIM_VALIDATION_FILEPATH = DATASET_FILEPATH / "validation_gensim.pkl"
LOCAL_TRAIN_FILEPATH = DATASET_FILEPATH / "train_local.pkl"
LOCAL_VALIDATION_FILEPATH = DATASET_FILEPATH / "validation_local.pkl"


EPOCHS = 2

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

    if use_gensim:
        with open(GENSIM_TRAIN_FILEPATH, "rb") as f:
            train_data = pickle.load(f)

        with open(GENSIM_VALIDATION_FILEPATH, "rb") as f:
            validation_data = pickle.load(f)
    else:
        with open(LOCAL_TRAIN_FILEPATH, "rb") as f:
            train_data = pickle.load(f)
        with open(LOCAL_VALIDATION_FILEPATH, "rb") as f:
            validation_data = pickle.load(f)

    total_train_len = math.ceil(len(train_data) / BATCH_SIZE)
    total_val_len = math.ceil(len(validation_data) / BATCH_SIZE)

    tokeniser = Tokeniser(use_gensim=use_gensim)

    model = TwoTowers(
        encoded_dim=encoded_dim,
        use_gensim=use_gensim,
        token_embed_dims=300 if use_gensim else 50,
        embed_layer_weights=(
            torch.load(W2V_EMBED_PATH, weights_only=True) if not use_gensim else None
        ),
    ).to(device)

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

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
            query, pos_samples, neg_samples = zip(*batch)
            loss, (pos_dist, neg_dist) = model.get_loss_batch(
                query, pos_samples, neg_samples
            )
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

        if not epoch % PERFORMANCE_EVAL_EVERY_N_EPOCHS:
            faiss_index = build_doc_faiss_index(model, tokeniser)

            hg_dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")
            train_score = evaluate_performance_two_towers(
                model=model,
                tokeniser=tokeniser,
                dataset_split=hg_dataset["train"][:200],
                faiss_index=faiss_index,
            )

            val_score = evaluate_performance_two_towers(
                model=model,
                tokeniser=tokeniser,
                dataset_split=hg_dataset["validation"],
                faiss_index=faiss_index,
            )
            wandb.log({"train_eval_score": train_score, "val_eval_score": val_score})
    wandb.finish()
