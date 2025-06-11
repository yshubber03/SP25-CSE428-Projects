# generated from chatgpt

import torch
import torch.nn as nn

class SmallSequenceCNN(nn.Module):
    def __init__(self, input_channels=4, embed_dim=768):
        super(SmallSequenceCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        # Input shape: (B, 4, seq_len)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.global_pool(x)  # (B, 256, 1)
        x = x.squeeze(-1)        # (B, 256)
        x = self.fc(x)           # (B, embed_dim)
        return x
