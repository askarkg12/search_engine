import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import random

from typing import TypeAlias

from pathlib import Path
import sys

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from utils.tokeniser import Tokeniser

TknSeq: TypeAlias = list[int]
Triplet: TypeAlias = tuple[TknSeq, TknSeq, TknSeq]
Pair: TypeAlias = tuple[TknSeq, TknSeq]


device = "cuda" if torch.cuda.is_available() else "cpu"
# map_location = torch.device(device)


class TripletDataset(Dataset):
    def __init__(
        self,
        dataset_split: dict,
        tokeniser: Tokeniser,
    ):
        # DO tokenising here
        # TODO
        query_doc_pairs = ...
        all_docs_tkns = ...

        self.query_doc_tkn_pairs = query_doc_pairs
        self.all_docs_tkns = all_docs_tkns

    def __len__(self):
        return len(self.query_doc_tkn_pairs)

    def __getitem__(self, idx):
        query: TknSeq
        pos_doc: TknSeq
        query, pos_doc = self.query_doc_tkn_pairs[idx]
        neg_doc: TknSeq = pos_doc
        while neg_doc == pos_doc:
            neg_doc = random.choice(self.all_docs_tkns)
        return query, pos_doc, neg_doc

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
        padded_query = pad_sequence(
            torch.tensor(query, dtype=torch.int32, requires_grad=False),
            batch_first=True,
        )
        padded_pos = pad_sequence(
            torch.tensor(pos, dtype=torch.int32, requires_grad=False),
            batch_first=True,
        )
        padded_neg = pad_sequence(
            torch.tensor(neg, dtype=torch.int32, requires_grad=False),
            batch_first=True,
        )
        return (
            (padded_query, query_lens),
            (padded_pos, pos_lens),
            (padded_neg, neg_lens),
        )


if __name__ == "__main__":
    dataset = TripletDataset(query_doc_pairs, all_docs_tkns)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch)
