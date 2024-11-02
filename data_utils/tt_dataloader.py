import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import random

from typing import TypeAlias

from tqdm import tqdm
from pathlib import Path
import sys

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from utils.tokeniser import Tokeniser

TknSeq: TypeAlias = list[int]
Triplet: TypeAlias = tuple[TknSeq, TknSeq, TknSeq]


class TripletDataset(Dataset):
    def __init__(
        self,
        dataset_split: dict,
        tokeniser: Tokeniser,
    ):
        # Create lookup set for all docs
        all_docs_str_set = set()
        passages = dataset_split["passages"]
        for passage in tqdm(passages):
            # At this point passage is a dict
            passage_texts = passage["passage_text"]
            all_docs_str_set.update(set(passage_texts))

        all_docs_strs = list(all_docs_str_set)
        doc_to_idx = {doc: idx for idx, doc in enumerate(all_docs_strs)}

        rows = tqdm(
            enumerate(dataset_split),
            total=len(dataset_split),
            desc=f"Tokenising",
        )
        query_doc_pairs: list[tuple[TknSeq, int]] = []

        for _, row in rows:
            query_tkns = tokeniser.tokenise_string(row["query"])
            pos_samples = row["passages"]["passage_text"]
            data = [
                (
                    query_tkns,
                    doc_to_idx[sample],
                )
                for sample in pos_samples
            ]
            query_doc_pairs.extend(data)

        self.query_doc_tkn_pairs = query_doc_pairs
        self.all_docs_tkns = [tokeniser.tokenise_string(doc) for doc in all_docs_strs]

    def __len__(self):
        return len(self.query_doc_tkn_pairs)

    def __getitem__(self, idx) -> Triplet:
        query, pos_doc_id = self.query_doc_tkn_pairs[idx]
        neg_doc_id = random.randint(0, len(self.all_docs_tkns) - 1)
        while neg_doc_id == pos_doc_id:
            neg_doc_id = random.randint(0, len(self.all_docs_tkns) - 1)
        return query, self.all_docs_tkns[pos_doc_id], self.all_docs_tkns[neg_doc_id]

    def collate_fn(self, batch: list[Triplet]):
        query, pos, neg = zip(*batch)

        # int16 is enough for the lengths, since max is defo less than 32k
        query_lens = torch.tensor(
            [len(q) for q in query], dtype=torch.int16, requires_grad=False
        )
        pos_lens = torch.tensor(
            [len(p) for p in pos], dtype=torch.int16, requires_grad=False
        )
        neg_lens = torch.tensor(
            [len(n) for n in neg], dtype=torch.int16, requires_grad=False
        )

        # int32 is enough for the token ids, since max is 3M
        query_tkn_seqs = [
            torch.tensor(q, dtype=torch.int32, requires_grad=False) for q in query
        ]
        pos_tkn_seqs = [
            torch.tensor(p, dtype=torch.int32, requires_grad=False) for p in pos
        ]
        neg_tkn_seqs = [
            torch.tensor(n, dtype=torch.int32, requires_grad=False) for n in neg
        ]
        padded_query = pad_sequence(query_tkn_seqs, batch_first=True)
        padded_pos = pad_sequence(pos_tkn_seqs, batch_first=True)
        padded_neg = pad_sequence(neg_tkn_seqs, batch_first=True)

        return (
            (padded_query, query_lens),
            (padded_pos, pos_lens),
            (padded_neg, neg_lens),
        )


if __name__ == "__main__":
    import datasets
    from torch.utils.data import DataLoader

    from utils.rich_utils import task

    with task("Initialising tokeniser"):
        tokeniser = Tokeniser(use_gensim=True)

    with task("Loading dataset"):
        dataset_split = datasets.load_dataset("microsoft/ms_marco", "v1.1")["test"]

    with task("Initialising dataset"):
        dataset_test = TripletDataset(dataset_split, tokeniser)

    test_dataloader = DataLoader(
        dataset_test, batch_size=16, collate_fn=dataset_test.collate_fn
    )

    for batch in test_dataloader:
        print(batch)
