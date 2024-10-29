import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import gensim.downloader as gs_api

device = "cuda" if torch.cuda.is_available() else "cpu"
map_location = torch.device(device)

from utils.tokeniser import Tokeniser

from typing import TypeAlias

Token: TypeAlias = int
Sequence: TypeAlias = list[Token]


class TwoTowers(nn.Module):
    def __init__(
        self,
        token_embed_dims: int,
        encoded_dim: int,
        use_gensim: bool = False,
        vocab_size: int = 81_547,
        tokeniser: Tokeniser | None = None,
        margin: float = 1.0,
        embed_layer_weights: torch.Tensor | None = None,
        freeze_embed_layer: bool = False,
        rnn_layer_num: int = 1,
    ):
        super().__init__()
        self.use_gensim = use_gensim
        self.margin = margin
        if use_gensim:
            self.token_embed_dims = 300
            self.embed_layer = gs_api.load("word2vec-google-news-300")

            def embed_gensim(text: str) -> torch.Tensor:
                # Returns list of vectors, shape [L, E]
                return torch.tensor(
                    [
                        (
                            self.embed_layer[word]
                            if word in self.embed_layer
                            else torch.zeros(300)
                        )
                        for word in text
                    ],
                    dtype=torch.float,
                    device=device,
                )

            self.embed_text = embed_gensim
        else:
            self.token_embed_dims = token_embed_dims
            if tokeniser is None:
                raise ValueError("tokeniser must be provided if use_gensim is False")
            self.tokeniser = tokeniser
            if embed_layer_weights is None:
                self.embed_layer = nn.Embedding(
                    num_embeddings=vocab_size, embedding_dim=self.token_embed_dims
                )
            else:
                self.embed_layer = nn.Embedding.from_pretrained(
                    embed_layer_weights, freeze=freeze_embed_layer
                )

            def embed_locally_trained(text: str) -> torch.Tensor:
                # Returns list of vectors, shape [L, E]
                return self.embed_layer(
                    # Returns list of ints, shape [L]
                    torch.tensor(
                        self.tokeniser.tokenise_string(text),
                        dtype=torch.long,
                        device=device,
                    )
                )

            self.embed_text = embed_locally_trained

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

    def encode_sequences(self, sequences: list[list[int]], encoder: nn.LSTM):
        seq_lens = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

        # Shape [N, Lrand, E]
        seq_embed_list = [self.embed_text(seq) for seq in sequences]

        # Shape [N, Lmax, E]
        padded_seq_embeds = pad_sequence(seq_embed_list, batch_first=True)

        packed_padded_seqs = pack_padded_sequence(
            padded_seq_embeds, seq_lens, batch_first=True, enforce_sorted=False
        )

        bi_seq_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_seq_encodings, _) = encoder(packed_padded_seqs)

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
    def forward(self, query: list[int]):
        return self.encode_query_single(query)


if __name__ == "__main__":
    pass
