import torch
import torch.nn as nn

from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pack_padded_sequence

from typing import TypeAlias

SequencesTensor: TypeAlias = torch.Tensor


class GenericEncoder(nn.Module):
    def __init__(self, embed_layer: nn.Module, encoding_layer: nn.Module):
        super().__init__()
        self.embed_layer: nn.Module = embed_layer
        self.encoding_layer: nn.Module = encoding_layer

    def forward(self, sequence: SequencesTensor, lengths: torch.Tensor) -> torch.Tensor:
        # Initial shapes: (batch_size, seq_len), (batch_size)

        # Shape: (batch_size, seq_len, embed_dim)
        seq_embeds = self.embed_layer(sequence)

        # Shape: (batch_size, hidden_dim)
        encoded_seqs = self.encoding_layer(seq_embeds, lengths)
        return encoded_seqs


class TokenEmbedder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings: torch.Tensor | None = None,
        vocab_size: int | None = None,
        embed_dim: int | None = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, sequence: SequencesTensor) -> torch.Tensor:
        return self.embedding(sequence)

    @staticmethod
    def from_gensim(model: KeyedVectors) -> "TokenEmbedder":
        embeddings = torch.tensor(model.vectors)
        return TokenEmbedder(pretrained_embeddings=embeddings)


class LSTMEncoder(nn.LSTM):
    def __init__(self, input_dims: int, encoded_size: int, bidirectional: bool = True):
        super().__init__(
            input_size=input_dims,
            hidden_size=encoded_size,
            num_layers=1,
            bidirectional=bidirectional,
        )
        self.bidirectional: bool = bidirectional

    def forward(
        self, sequences: SequencesTensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        encoded_seqs: torch.Tensor

        # NOTE: I wonder if there are going to be issues with device
        packed_seq_embeds = pack_padded_sequence(
            sequences, lengths, batch_first=True, enforce_sorted=False
        )

        if self.bidirectional:
            _, (encoded_seqs, _) = self(packed_seq_embeds)
            batch_len = encoded_seqs.shape[1]
            encoded_seqs = encoded_seqs.permute(1, 0, 2).reshape(batch_len, -1)
        else:
            _, encoded_seqs = self(packed_seq_embeds)

        return encoded_seqs


class PoolingEncoder(nn.Module):
    def __init__(self, embedding_size: int, encoded_size: int, num_layers: int = 1):
        super().__init__()
        layers = []

        # I want to basically gradually shift the Linear layer sizes
        step = (embedding_size - encoded_size) / num_layers
        output_dims = [embedding_size + int(step * i) for i in range(1, num_layers + 1)]
        input_dims = [encoded_size] + output_dims[:-1]

        for input_dim, output_dim in zip(input_dims[:-1], output_dims[:-1]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(input_dims[-1], embedding_size))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, embedded_seqs: SequencesTensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # Shape: (batch_size, seq_len, embed_dim)
        batch_size = embedded_seqs.shape[0]

        # List of (lengths[i], embed_dim)
        unpadded_seqs = [embedded_seqs[i, : lengths[i]] for i in range(batch_size)]

        # Shape: (batch_size, embed_dim)
        pooled_embeddings = torch.tensor(
            # Shape: (lengths[i], embed_dim)
            [sequence.mean(dim=0) for sequence in unpadded_seqs]
        )

        return self.mlp(pooled_embeddings)
