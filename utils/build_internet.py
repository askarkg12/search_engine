import datasets
import pickle
from tqdm import tqdm
from pathlib import Path

INTERNET_FILEPATH = Path(f"dataset/internet/all_docs")
INTERNET_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

    splits = ["train", "test", "validation"]
    all_documents: set[str] = set()

    for split in splits:

        passages = dataset[split]["passages"]  # [:10000]
        for passage in tqdm(passages, desc=split + " passages"):
            all_documents.update(set(passage["passage_text"]))

    with open(INTERNET_FILEPATH, "w", encoding="utf-8") as f:
        for index, doc in tqdm(
            enumerate(all_documents),
            desc="Writing into document",
            total=len(all_documents),
        ):
            f.write(f"{doc}\n")
