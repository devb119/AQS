import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(in_features, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.fc3 = nn.Linear(num_hidden_units, num_hidden_units)
        self.fc4 = nn.Linear(num_hidden_units, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x