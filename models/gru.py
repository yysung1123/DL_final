

from torch import nn
import torch
import torch.nn.functional as F


class GRUNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False, num_layers=2, dropout=0.5)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, label_size)


    def forward(self, input, input_lens, hidden=None):
        sorted_lens, indices = torch.sort(input_lens, descending=True)
        input = input[indices]
        _, desorted_indices = torch.sort(indices, descending=False)
        embedded = self.word_embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lens, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[desorted_indices]
        outputs = torch.sum(outputs, dim=1)
        for idx in range(len(input_lens)):
            outputs[idx] /= input_lens[idx].float()
        out = F.leaky_relu(self.fc1(outputs))
        out = self.fc2(out)
        return out

