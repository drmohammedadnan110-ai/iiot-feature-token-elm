"""Feature-specific numerical/category tokens, identity embeddings and self-attention."""

import torch
from torch import nn
from torch.nn import functional as F


class FeatureTokenizer(nn.Module):
    def __init__(self, n_numeric, cardinalities, width):
        super().__init__()
        self.n_numeric = n_numeric
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, width))
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, width))
        self.categories = nn.ModuleList([nn.Embedding(k, width) for k in cardinalities])
        self.identity = nn.Parameter(torch.empty(n_numeric + len(cardinalities), width))
        nn.init.normal_(self.numeric_weight, std=0.02)
        nn.init.normal_(self.identity, std=0.02)
        for embedding in self.categories:
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, numeric, categorical):
        tokens = [numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias]
        tokens += [embedding(categorical[:, i]).unsqueeze(1) for i, embedding in enumerate(self.categories)]
        return torch.cat(tokens, dim=1) + self.identity


class FeatureEncoder(nn.Module):
    """Explicit reference choice: pre-LN residual blocks and mean pooling (no CLS)."""

    def __init__(
        self, n_numeric, cardinalities, width=96, layers=2, heads=4, expansion=3, dropout=0.1, attention=True
    ):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid encoder dimensions/dropout")
        if n_numeric + len(cardinalities) < 1:
            raise ValueError("At least one predictor is required")
        self.width = width
        self.tokenizer = FeatureTokenizer(n_numeric, cardinalities, width)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if attention:
                block = nn.TransformerEncoderLayer(
                    width,
                    heads,
                    int(width * expansion),
                    dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            else:
                block = TokenFeedForward(width, int(width * expansion), dropout)
            self.blocks.append(block)
        self.norm = nn.LayerNorm(width)

    def forward(self, numeric, categorical):
        tokens = self.tokenizer(numeric, categorical)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).mean(1)


class TokenFeedForward(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class TrainingModel(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.width, classes)

    def forward(self, numeric, categorical):
        return self.head(self.encoder(numeric, categorical))


def inverse_frequency_weights(labels, classes):
    counts = torch.bincount(torch.as_tensor(labels), minlength=classes).float()
    if (counts == 0).any():
        raise ValueError("Every task class must occur in TRAIN")
    return counts.sum() / (classes * counts)


def focal_loss(logits, labels, weights, gamma=2.0):
    if gamma < 0:
        raise ValueError("Focal gamma cannot be negative")
    log_pt = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    return (-weights[labels] * (1 - log_pt.exp()).pow(gamma) * log_pt).mean()
