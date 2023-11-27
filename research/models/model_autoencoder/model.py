import torch.nn as nn

"""This is a auto-encoder model that takes in the bitboard as input along with
some other heuristics."""

class ChessAutoEncoder(nn.Module):
    def __init__(self, feature_length):
        super(ChessAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_length, feature_length//2),
            nn.ReLU(),
            nn.Linear(feature_length//2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Compressed representation
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x