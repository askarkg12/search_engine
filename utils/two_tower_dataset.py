import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from typing import TypeAlias

Sequence: TypeAlias = list[int]


def collate_fn(
    batch: list[tuple[Sequence, Sequence, Sequence]]
) -> tuple[Sequence, Sequence, Sequence]:
    queries, pos, negs = zip(*batch)

    queries = [torch.tensor(q) for q in queries]
    pos = [torch.tensor(p) for p in pos]
    negs = [torch.tensor(n) for n in negs]

    queries = pad_sequence(queries, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    negs = pad_sequence(negs, batch_first=True, padding_value=0)
    return queries, pos, negs


class TwoTowerDataset(Dataset):
    def __init__(self, data: list[tuple[Sequence, Sequence, Sequence]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Sequence, Sequence, Sequence]:
        return self.data[idx]
