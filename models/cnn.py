

from torch import nn
import torch
import torch.nn.functional as F


class TCNN(nn.Module):
    def __init__(self, vocab_size, label_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, 64)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.avgpool1 = nn.AvgPool1d(64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, label_size)

    def forward(self, x):
        out = self.word_embedding(x)
        out = self.conv1(out)
        out = self.avgpool1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
