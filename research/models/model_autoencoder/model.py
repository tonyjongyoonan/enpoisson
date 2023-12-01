import torch.nn as nn
import torch

"""This is a auto-encoder model that takes in the bitboard as input along with
some other heuristics."""

class ChessAutoEncoder(nn.Module):
    def __init__(self, feature_length):
        super(ChessAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 50)  # Compressed representation
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(101, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        original_board = self.encoder(x[:,1:769])
        new_board = self.encoder(x[:,769:1537])
        x = self.classifier(torch.cat((x[:,0:1],original_board,new_board),dim=1))
        return x