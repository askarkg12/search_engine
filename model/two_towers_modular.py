import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import gensim.downloader as gs_api
from gensim.models import KeyedVectors

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.tokeniser import Tokeniser

from typing import TypeAlias

Token: TypeAlias = int
Sequence: TypeAlias = list[Token]
SequenceTensor: TypeAlias = torch.Tensor


class TwoTowers(nn.Module):
    def __init__(
        self,
        token_embed_dims: int,
        encoded_dim: int,
        use_gensim: bool = False,
        vocab_size: int = 81_547,
        margin: float = 1.0,
        embed_layer_weights: torch.Tensor | None = None,
        freeze_embed_layer: bool = False,
        rnn_layer_num: int = 1,
    ):
        super().__init__()
        self.margin = margin

        # Embedding layer can be external or internal
        if use_gensim:
            w2v: KeyedVectors = gs_api.load("word2vec-google-news-300")
            embeddings = w2v.vectors
            self.embed_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings), freeze=freeze_embed_layer
            ).to(device)
        else:
            if embed_layer_weights is None:
                self.embed_layer = nn.Embedding(
                    num_embeddings=vocab_size, embedding_dim=token_embed_dims
                ).to(device)
            else:
                self.embed_layer = nn.Embedding.from_pretrained(
                    embed_layer_weights, freeze=freeze_embed_layer
                ).to(device)

        self.query_encoder = nn.LSTM(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.doc_encoder = nn.LSTM(
            input_size=token_embed_dims,
            hidden_size=encoded_dim,
            num_layers=rnn_layer_num,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(
        #     distance_function=self.dist_function_single, reduction="mean"
        # )

    def dist_function_single(self, query, sample):
        # Shape [N, 2*H]
        return 1 - nn.functional.cosine_similarity(query, sample, dim=1)

    def get_loss_batch(
        self,
        query_tkns: list[Sequence],
        positive_tkns: list[Sequence],
        negative_tkns: list[Sequence],
    ):
        query_encodings = self.encode_queries(query_tkns)
        pos_encodings = self.encode_docs(positive_tkns)
        neg_encodings = self.encode_docs(negative_tkns)

        pos_dist = self.dist_function_single(query_encodings, pos_encodings)
        neg_dist = self.dist_function_single(query_encodings, neg_encodings)

        triplet_loss = torch.max(pos_dist - neg_dist + self.margin, torch.tensor(0.0))

        return triplet_loss.mean(), (pos_dist.mean(), neg_dist.mean())

    def encode_sequences(self, token_sequences: list[SequenceTensor], encoder: nn.LSTM):
        # Shape [N, Lrand] at the start

        # Shape [N]
        seq_lens = torch.tensor(
            [seq.shape[0] for seq in token_sequences],
            dtype=torch.long,
            device="cpu",
        )

        # Shape [N, Lmax]
        padded_seq_tkns = pad_sequence(token_sequences, batch_first=True).to(device)

        # Shape [N, Lmax, E]
        padded_seq_embeds = self.embed_layer(padded_seq_tkns)

        # Shape [N, Lmax, E]
        packed_padded_seq_embeds = pack_padded_sequence(
            padded_seq_embeds, seq_lens, batch_first=True, enforce_sorted=False
        )

        bi_seq_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_seq_encodings, _) = encoder(packed_padded_seq_embeds)

        batch_len = bi_seq_encodings.shape[1]

        # Convert to [N, 2*H]
        seq_encodings = bi_seq_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        return seq_encodings

    def encode_docs(self, docs: list[str]):
        return self.encode_sequences(docs, self.doc_encoder)

    def encode_queries(self, queries: list[str]):
        return self.encode_sequences(queries, self.query_encoder)

    def encode_sequence_single(self, sequence: list[int], encoder: nn.LSTM):
        return self.encode_sequences([sequence], encoder)[0]

    def encode_query_single(self, query: str):
        return self.encode_sequence_single(query, self.query_encoder)

    def encode_doc_single(self, doc: str):
        return self.encode_sequence_single(doc, self.doc_encoder)

    # Forward because model is used in app
    def forward(self, query: list[int]) -> torch.Tensor:
        return self.encode_query_single(query)

    def save_without_embed_layer(self, filepath: Path):
        state_dict = self.state_dict()
        del state_dict["embed_layer.weight"]
        torch.save(state_dict, filepath)


def load_two_tower_without_embed(filepath: Path, **kwargs):
    model = TwoTowers(**kwargs)
    model.load_state_dict(torch.load(filepath), strict=False)
    return model


if __name__ == "__main__":
    model = TwoTowers(encoded_dim=400, use_gensim=True)
    save_path = root_dir / "weights/aaa.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_without_embed_layer(save_path)
