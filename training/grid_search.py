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
from utils.rich_utils import task
from training.performance_eval import (
    evaluate_performance_two_towers,
    build_doc_faiss_index,
)

BATCH_SIZE = 1024

PERFORMANCE_EVAL_EVERY_N_EPOCHS = 5


W2V_EMBED_PATH = root_dir / "weights/w2v_embeddings.pth"

DATASET_FILEPATH = root_dir / "dataset/two_tower/prep"

USE_MINI_DATASET = False

mini_option = "_mini" if USE_MINI_DATASET else ""

GENSIM_TRAIN_FILEPATH = DATASET_FILEPATH / f"train_gensim_tensors{mini_option}.pkl"
GENSIM_VALIDATION_FILEPATH = (
    DATASET_FILEPATH / f"validation_gensim_tensors{mini_option}.pkl"
)
LOCAL_TRAIN_FILEPATH = DATASET_FILEPATH / f"train_local_tensors{mini_option}.pkl"
LOCAL_VALIDATION_FILEPATH = (
    DATASET_FILEPATH / f"validation_local_tensors{mini_option}.pkl"
)


EPOCHS = 30

MODEL_CONFIGS = [
    {
        "run_name": "gensim_300",
        "use_gensim": True,
        "optimizer": "adam",
        "lr": 0.001,
        "encoded_dim": 300,
    },
    {
        "run_name": "gensim_400",
        "use_gensim": True,
        "optimizer": "adam",
        "lr": 0.001,
        "encoded_dim": 400,
    },
    {
        "run_name": "gensim_500",
        "use_gensim": True,
        "optimizer": "adam",
        "lr": 0.001,
        "encoded_dim": 500,
    },
    {
        "run_name": "no_gensim_200",
        "use_gensim": False,
        "encoded_dim": 200,
        "optimizer": "adam",
        "lr": 0.001,
    },
    {
        "run_name": "no_gensim_300",
        "use_gensim": False,
        "encoded_dim": 300,
        "optimizer": "adam",
        "lr": 0.001,
    },
    {
        "run_name": "no_gensim_400",
        "use_gensim": False,
        "encoded_dim": 400,
        "optimizer": "adam",
        "lr": 0.001,
    },
]

for config in MODEL_CONFIGS:
    run_name = config["run_name"] + mini_option
    use_gensim = config["use_gensim"]
    encoded_dim = config["encoded_dim"]
    optimizer = config["optimizer"]
    lr = config["lr"]

    if use_gensim:
        with task(f"Loading {run_name} train data"):
            with open(GENSIM_TRAIN_FILEPATH, "rb") as f:
                train_data = pickle.load(f)

        with task(f"Loading {run_name} validation data"):
            with open(GENSIM_VALIDATION_FILEPATH, "rb") as f:
                validation_data = pickle.load(f)
    else:
        with task(f"Loading {run_name} train data"):
            with open(LOCAL_TRAIN_FILEPATH, "rb") as f:
                train_data = pickle.load(f)

        with task(f"Loading {run_name} validation data"):
            with open(LOCAL_VALIDATION_FILEPATH, "rb") as f:
                validation_data = pickle.load(f)

    total_train_len = math.ceil(len(train_data) / BATCH_SIZE)
    total_val_len = math.ceil(len(validation_data) / BATCH_SIZE)

    with task(f"Initialising {run_name} tokeniser"):
        tokeniser = Tokeniser(use_gensim=use_gensim)

    with task(f"Initialising {run_name} model"):
        model = TwoTowers(
            encoded_dim=encoded_dim,
            use_gensim=use_gensim,
            token_embed_dims=300 if use_gensim else 50,
            embed_layer_weights=(
                torch.load(W2V_EMBED_PATH, weights_only=True)
                if not use_gensim
                else None
            ),
        ).to(device)

    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    wandb.init(project="two_tower_grid_search_epochs", name=run_name, config=config)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_chunks = more_itertools.chunked(train_data, BATCH_SIZE)
        loss_list = []
        pos_dist_list = []
        neg_dist_list = []
        for batch in tqdm(
            train_chunks,
            desc=f"{epoch} - training",
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

        # Validation
        model.eval()
        validation_chunks = more_itertools.chunked(validation_data, BATCH_SIZE)
        val_loss_list = []
        val_pos_dist_list = []
        val_neg_dist_list = []
        with torch.inference_mode():
            for batch in tqdm(
                validation_chunks,
                desc=f"{epoch} - validation losses",
                total=total_val_len,
                unit_scale=BATCH_SIZE,
            ):
                query, pos_samples, neg_samples = zip(*batch)
                loss, (pos_dist, neg_dist) = model.get_loss_batch(
                    query, pos_samples, neg_samples
                )
                val_loss_list.append(loss.item())
                val_pos_dist_list.append(pos_dist.item())
                val_neg_dist_list.append(neg_dist.item())

        wandb.log(
            {
                "train_loss": sum(loss_list) / len(loss_list),
                "train_pos_dist": sum(pos_dist_list) / len(pos_dist_list),
                "train_neg_dist": sum(neg_dist_list) / len(neg_dist_list),
                "val_loss": sum(val_loss_list) / len(val_loss_list),
                "val_pos_dist": sum(val_pos_dist_list) / len(val_pos_dist_list),
                "val_neg_dist": sum(val_neg_dist_list) / len(val_neg_dist_list),
                "epoch": epoch,
            },
        )

        if not epoch % PERFORMANCE_EVAL_EVERY_N_EPOCHS:
            faiss_index = build_doc_faiss_index(model, tokeniser)

            hg_dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")
            train_score = evaluate_performance_two_towers(
                model=model,
                tokeniser=tokeniser,
                dataset_split=hg_dataset["train"][:100],
                faiss_index=faiss_index,
            )

            val_score = evaluate_performance_two_towers(
                model=model,
                tokeniser=tokeniser,
                dataset_split=hg_dataset["validation"][:100],
                faiss_index=faiss_index,
            )
            wandb.log(
                {
                    "train_eval_score": float(train_score),
                    "val_eval_score": float(val_score),
                    "epoch": epoch,
                },
            )

            # Save model
            save_path = root_dir / f"weights/checkpoints/two_towers_{run_name}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_without_embed_layer(save_path)

            # Send artifact to wandb
            artifact = wandb.Artifact(
                f"two_towers_{run_name}", type="model", metadata=config
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

    wandb.finish()
