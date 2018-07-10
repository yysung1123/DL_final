

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch
import json


from dataset import Corpus, TextDataset, choose_chapters1, choose_chapters2
from models.cnn import TCNN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)


label_size = 8
batch_size = 64
learning_rate = 0.001
epochs = 40
chapters = choose_chapters1()
cp = Corpus(chapters)
train_set = TextDataset(cp, train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = TextDataset(cp, train=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
model = TCNN(vocab_size=len(cp.vocab), label_size=label_size).cuda()
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()

record = {}
record['train_loss'] = []
record['test_loss'] = []
record['test_correct'] = []
record['confusion'] = []
record['chapters'] = chapters

for epoch in range(epochs):
    current_loss = 0
    model.train()
    for data, target, lens in train_loader:
        data = data.cuda()
        target = target.cuda()
        lens = lens.cuda()
        model.zero_grad()
        model.batch_size = len(data)
        output = model(data)
        loss = loss_function(output, target.view(-1))
        loss.backward()
        current_loss += loss.item()
        optimizer.step()
    current_loss /= len(train_loader)
    print("Epoch {}: Train loss: {}".format(epoch, current_loss))
    record['train_loss'].append(current_loss)

    current_loss = 0
    correct = 0
    model.eval()
    cm = torch.zeros((label_size, label_size))
    for data, target, lens in test_loader:
        data = data.cuda()
        target = target.cuda()
        lens = lens.cuda()
        model.batch_size = len(data)
        output = model(data)
        loss = loss_function(output, target.view(-1))
        current_loss += loss.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view(-1)).cpu().sum()
        for idx in range(len(pred)):
            cm[target[idx].item()][pred[idx].item()] += 1
    current_loss /= len(test_loader)
    print("Epoch {}: Test loss: {}".format(epoch, current_loss))
    print("Epoch {}: Test correct: {}/{} {}%".format(epoch, correct, len(test_loader.dataset), correct * 100 / len(test_loader.dataset)))
    print(cm.data.numpy())
    cm = [[cm[:4, :4].sum().item(), cm[:4, 4:].sum().item()], [cm[4:, :4].sum().item(), cm[4:, 4:].sum().item()]]
    print(cm)
    record['test_loss'].append(current_loss)
    record['test_correct'].append((correct * 100 / len(test_loader.dataset)).item())
    record['confusion'].append(cm)

with open('result.json', 'w') as f:
    json.dump(record, f)
