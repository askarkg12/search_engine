import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 50


class Word2Vec(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embedding_dim)
        self.context_projection_embed = nn.Embedding(vocab_size, embedding_dim)
        # self.sig = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def get_loss(self, inpt, trgs, rand):
        emb: torch.Tensor = self.center_embed(inpt)

        ctx = self.context_projection_embed(trgs)
        neg = self.context_projection_embed(rand)

        pos_logits = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
        neg_logits = torch.bmm(neg, emb.unsqueeze(-1)).squeeze()

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        return self.loss(logits, labels)

    def forward(self, id):
        return self.center_embed(id)
