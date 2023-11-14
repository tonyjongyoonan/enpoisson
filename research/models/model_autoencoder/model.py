import torch
import torch.nn as nn
import torch.nn.functional as F

"""This is a auto-encoder model that takes in the bitboard as input along with
some other heuristics."""

class ChessAutoEncoder(nn.Module):
    def __init__(self):
        super(ChessAutoEncoder, self).__init__()
        # Assuming each channel represents a different piece type (e.g., 12 channels for 6 types each for black and white)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming an 8x8 chess board
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Using sigmoid for binary classification

        return x