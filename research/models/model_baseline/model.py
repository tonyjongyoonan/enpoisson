import torch
import torch.nn as nn
import torch.nn.functional as F

"""This is a baseline model that takes in hueristics as 
features of the board state as our main input."""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=INPUT_FEATURES, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x