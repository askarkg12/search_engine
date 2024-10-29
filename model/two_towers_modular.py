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
        vocab_size,
        token_embed_dims,
        encoded_dim,
        rnn_layer_num=1,
        use_gensim=False,
        tokeniser: Tokeniser | None = None,
    ):
        super().__init__()
        self.use_gensim = use_gensim
        if use_gensim:
            self.embed_layer = gs_api.load("word2vec-google-news-300")

            def embed_gensim(text):
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
            if tokeniser is None:
                raise ValueError("tokeniser must be provided if use_gensim is False")
            self.tokeniser = tokeniser
            self.embed_layer = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=token_embed_dims
            )

            def embed_locally_trained(text: str):
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
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_function_single, reduction="mean"
        )

    def dist_function_single(self, query, sample):
        # Shape [N, 2*H]
        return 1 - nn.functional.cosine_similarity(query, sample, dim=1)

    def get_loss_batch_with_distances(
        self,
        query_tkns: list[Sequence],
        positive_tkns: list[Sequence],
        negative_tkns: list[Sequence],
    ):
        # Shape [N]
        query_lengths = torch.tensor([len(seq) for seq in query_tkns], dtype=torch.long)
        pos_lengths = torch.tensor(
            [len(seq) for seq in positive_tkns], dtype=torch.long
        )
        neg_lengths = torch.tensor(
            [len(seq) for seq in negative_tkns], dtype=torch.long
        )

        query_tkns = [
            torch.tensor(q, dtype=torch.long, device=device) for q in query_tkns
        ]
        positive_tkns = [
            torch.tensor(p, dtype=torch.long, device=device) for p in positive_tkns
        ]
        negative_tkns = [
            torch.tensor(n, dtype=torch.long, device=device) for n in negative_tkns
        ]

        # Shape [N, Lmax] (for each)
        padded_query_tkns = pad_sequence(query_tkns, batch_first=True)
        padded_pos_tkns = pad_sequence(positive_tkns, batch_first=True)
        padded_neg_tkns = pad_sequence(negative_tkns, batch_first=True)

        # Shape [N, Lmax, E]
        padded_query_embeds = self.embed_text(padded_query_tkns)
        padded_pos_embeds = self.embed_text(padded_pos_tkns)
        padded_neg_embeds = self.embed_text(padded_neg_tkns)

        packed_padded_queries = pack_padded_sequence(
            padded_query_embeds, query_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_pos = pack_padded_sequence(
            padded_pos_embeds, pos_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_neg = pack_padded_sequence(
            padded_neg_embeds, neg_lengths, batch_first=True, enforce_sorted=False
        )

        bi_query_encodings: torch.Tensor
        bi_pos_encodings: torch.Tensor
        bi_neg_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_query_encodings, _) = self.query_encoder(packed_padded_queries)
        _, (bi_pos_encodings, _) = self.doc_encoder(packed_padded_pos)
        _, (bi_neg_encodings, _) = self.doc_encoder(packed_padded_neg)

        batch_len = bi_query_encodings.shape[1]

        # Convert to [N, 2*H]
        query_encodings = bi_query_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        pos_encodings = bi_pos_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        neg_encodings = bi_neg_encodings.permute(1, 0, 2).reshape(batch_len, -1)

        loss = self.triplet_loss(query_encodings, pos_encodings, neg_encodings)

        # Shape [N]
        pos_distances = self.dist_function_single(query_encodings, pos_encodings).mean()
        neg_distances = self.dist_function_single(query_encodings, neg_encodings).mean()
        return loss, pos_distances, neg_distances

    def get_loss_batch(
        self,
        query_tkns: list[Sequence],
        positive_tkns: list[Sequence],
        negative_tkns: list[Sequence],
    ):
        # Shape [N]
        query_lengths = torch.tensor([len(seq) for seq in query_tkns], dtype=torch.long)
        pos_lengths = torch.tensor(
            [len(seq) for seq in positive_tkns], dtype=torch.long
        )
        neg_lengths = torch.tensor(
            [len(seq) for seq in negative_tkns], dtype=torch.long
        )

        query_tkns = [
            torch.tensor(q, dtype=torch.long, device=device) for q in query_tkns
        ]
        positive_tkns = [
            torch.tensor(p, dtype=torch.long, device=device) for p in positive_tkns
        ]
        negative_tkns = [
            torch.tensor(n, dtype=torch.long, device=device) for n in negative_tkns
        ]

        # Shape [N, Lmax] (for each)
        padded_query_tkns = pad_sequence(query_tkns, batch_first=True)
        padded_pos_tkns = pad_sequence(positive_tkns, batch_first=True)
        padded_neg_tkns = pad_sequence(negative_tkns, batch_first=True)

        # Shape [N, Lmax, E]
        padded_query_embeds = self.embed_text(padded_query_tkns)
        padded_pos_embeds = self.embed_text(padded_pos_tkns)
        padded_neg_embeds = self.embed_text(padded_neg_tkns)

        packed_padded_queries = pack_padded_sequence(
            padded_query_embeds, query_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_pos = pack_padded_sequence(
            padded_pos_embeds, pos_lengths, batch_first=True, enforce_sorted=False
        )
        packed_padded_neg = pack_padded_sequence(
            padded_neg_embeds, neg_lengths, batch_first=True, enforce_sorted=False
        )

        bi_query_encodings: torch.Tensor
        bi_pos_encodings: torch.Tensor
        bi_neg_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_query_encodings, _) = self.query_encoder(packed_padded_queries)
        _, (bi_pos_encodings, _) = self.doc_encoder(packed_padded_pos)
        _, (bi_neg_encodings, _) = self.doc_encoder(packed_padded_neg)

        batch_len = bi_query_encodings.shape[1]

        # Convert to [N, 2*H]
        query_encodings = bi_query_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        pos_encodings = bi_pos_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        neg_encodings = bi_neg_encodings.permute(1, 0, 2).reshape(batch_len, -1)

        return self.triplet_loss(query_encodings, pos_encodings, neg_encodings)

    # Forward because model is used in app
    def forward(self, query: list[int]):
        return self.encode_query_single(query)

    def encode_query_single(self, query: str):

        # Shape [L, E]
        query_embed = self.embed_text(query)
        # Shape [2, H]
        query_rnn_embed: torch.Tensor
        _, (query_rnn_embed, _) = self.query_encoder(query_embed)
        # Shape [H]
        return query_rnn_embed.reshape(-1)

    def encode_doc_single(self, doc: list[int]):
        doc_tensor = torch.tensor(doc, dtype=torch.long, device=device)
        # Shape [L, E]
        doc_embed = self.embed_text(doc_tensor)
        # Shape [2, H]
        doc_rnn_embed: torch.Tensor
        _, (doc_rnn_embed, _) = self.doc_encoder(doc_embed)
        # Shape [H]
        return doc_rnn_embed.reshape(-1)

    def encode_docs(self, docs: list[list[int]]):
        # Shape [N]
        doc_lengths = torch.tensor([len(seq) for seq in docs], dtype=torch.long)

        docs = [torch.tensor(d, dtype=torch.long, device=device) for d in docs]

        # Shape [N, Lmax] (for each)
        padded_docs = pad_sequence(docs, batch_first=True)

        # Shape [N, Lmax, E]
        padded_docs_embeds = self.embed_text(padded_docs)

        packed_padded_docs = pack_padded_sequence(
            padded_docs_embeds, doc_lengths, batch_first=True, enforce_sorted=False
        )

        bi_docs_encodings: torch.Tensor
        # Shape [2, N, H]
        _, (bi_docs_encodings, _) = self.doc_encoder(packed_padded_docs)

        batch_len = bi_docs_encodings.shape[1]

        # Convert to [N, 2*H]
        docs_encodings = bi_docs_encodings.permute(1, 0, 2).reshape(batch_len, -1)
        return docs_encodings


if __name__ == "__main__":
    model = TwoTowers(200, 10, 20)

    batch_num = 20

    que_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]
    pos_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]
    neg_list = [
        torch.randint(0, 200, [torch.randint(5, 15, ())]) for _ in range(batch_num)
    ]

    loss = model.get_loss_batch(
        query_tkns=que_list, positive_tkns=pos_list, negative_tkns=neg_list
    )
    pass
