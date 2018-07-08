

from torch import nn
import torch


class LSTMNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, label_size)


    def forward(self, input, input_lens, hidden=None):
        sorted_lens, indices = torch.sort(input_lens, descending=True)
        input = input[indices]
        _, desorted_indices = torch.sort(indices, descending=False)
        embedded = self.word_embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lens, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[desorted_indices]
        row_indices = torch.arange(0, self.batch_size).long()
        col_indices = input_lens - 1
        row_indices = row_indices.cuda()
        col_indices = col_indices.cuda()
        last_tensor = outputs[row_indices, col_indices, :]
        out = self.bn1(last_tensor)
        out = self.fc1(out)
        return out
