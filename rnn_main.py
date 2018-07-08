
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch
import json


from dataset import Corpus, TextDataset, TextDataLoader
from models.lstm import LSTMNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'LSTM':
        nn.init.orthogonal_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)


cp = Corpus()
train_set = TextDataset(cp, train=True)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = TextDataset(cp, train=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
learning_rate = 0.005
epochs = 100

model = LSTMNet(50, 64, vocab_size=len(cp.vocab), label_size=3, batch_size=64).cuda()
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()

record = {}
record['train_loss'] = []
record['test_loss'] = []

for epoch in range(epochs):
    current_loss = 0
    model.train()
    for data, target, lens in train_loader:
        data = data.cuda()
        target = target.cuda()
        lens = lens.cuda()
        model.zero_grad()
        model.batch_size = len(data)
        output = model(data, lens)
        loss = loss_function(output, target.view(-1))
        loss.backward()
        current_loss += loss.item()
        optimizer.step()
    current_loss /= len(train_loader)
    print("Epoch {}: Train loss: {}".format(epoch, current_loss))
    record['train_loss'].append(current_loss)

    current_loss = 0
    model.eval()
    for data, target, lens in test_loader:
        data = data.cuda()
        target = target.cuda()
        lens = lens.cuda()
        model.batch_size = len(data)
        output = model(data, lens)
        loss = loss_function(output, target.view(-1))
        current_loss += loss.item()
    current_loss /= len(test_loader)
    print("Epoch {}: Test loss: {}".format(epoch, current_loss))
    record['test_loss'].append(current_loss)

with open('result.json', 'w') as f:
    json.dump(record, f)
