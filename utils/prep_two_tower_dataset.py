import datasets
import pickle
from tqdm import tqdm
from pathlib import Path
from tokeniser import Tokeniser
from random import randrange
import sys

SAVE_AS_TENSORS = True
if SAVE_AS_TENSORS:
    import torch

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from rich_utils import task

DATASET_FOLDER = root_dir / "dataset" / "two_tower" / "prep"
DATASET_FOLDER.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    with task(f"Loading dataset"):
        dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

    splits = ["test", "validation", "train"]
    all_documents: set[str] = set()

    for split in splits:

        passages = dataset[split]["passages"]
        for passage in tqdm(passages, desc=split + " passages"):
            all_documents.update(set(passage["passage_text"]))

    with task("Creating document list"):
        all_documents = list(all_documents)
        internet_length = len(all_documents)

    for TOKENISER in ["gensim", "local"]:
        with task(f"Initialising tokeniser ({TOKENISER})"):
            tokeniser = Tokeniser(use_gensim=TOKENISER == "gensim")
            tknz = tokeniser.tokenise_string

        for split in splits:
            split_data = []
            split_data_tensors = []
            rows = tqdm(
                enumerate(dataset[split]),
                total=dataset[split].num_rows,
                desc=f"Tokenising {split} ({TOKENISER})",
            )
            for row_index, row in rows:
                query_tkns = tknz(row["query"])
                pos_samples = row["passages"]["passage_text"]
                data = [
                    (
                        query_tkns,
                        tknz(sample),
                        tknz(all_documents[randrange(internet_length)]),
                    )
                    for sample in pos_samples
                ]
                if SAVE_AS_TENSORS:
                    data_tensors = [
                        (torch.tensor(query), torch.tensor(pos), torch.tensor(neg))
                        for query, pos, neg in data
                    ]

                split_data.extend(data)
                split_data_tensors.extend(data_tensors)

            with task(f"Saving {split} integers ({TOKENISER})"):
                with open(DATASET_FOLDER / f"{split}_{TOKENISER}.pkl", "wb") as f:
                    pickle.dump(split_data, f)

            if SAVE_AS_TENSORS:
                # Save a mini version for testing
                with task(f"Saving {split} tensors mini ({TOKENISER})"):
                    with open(
                        DATASET_FOLDER / f"{split}_{TOKENISER}_tensors_mini.pkl", "wb"
                    ) as f:
                        pickle.dump(split_data_tensors[:10000], f)

                with task(f"Saving {split} tensors ({TOKENISER})"):
                    with open(
                        DATASET_FOLDER / f"{split}_{TOKENISER}_tensors.pkl", "wb"
                    ) as f:
                        pickle.dump(split_data_tensors, f)
