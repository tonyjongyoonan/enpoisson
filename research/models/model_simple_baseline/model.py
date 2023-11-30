import torch.nn as nn

"""This is a auto-encoder model that takes in the bitboard as input along with
some other heuristics."""

class ChessAutoEncoder(nn.Module):
    def __init__(self, feature_length):
        super(ChessAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_length, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100)  # Compressed representation
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(201, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        original_board = self.encoder(x[1:769])
        new_board = self.encoder(x[769:1537])
        x = self.classifier([x[0]] + original_board + new_board)
        return x