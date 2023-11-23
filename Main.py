import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Dataset import MyDataset

dataset = MyDataset(file="./resources/annots.csv", root_dir='./resources/imgs', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [3, 2])
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

x = list()
y = list()
for t in train_set:
    x.append(t[0])
    y.append(t[1])

x_train = torch.stack(x)
y_train = torch.tensor(y)

x = list()
y = list()
for t in test_set:
    x.append(t[0])
    y.append(t[1])

x_test = torch.stack(x)
y_test = torch.tensor(y)

x_train.reshape([-1, 28, 28])
x_test.reshape([-1, 28, 28])

class GoodMorningNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(GoodMorningNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_neurons)
        self.activ1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_neurons, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        return x

net = GoodMorningNet(50)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = net.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1.0e-3)

batch_size = 100

x_test.to(device)
y_test.to(device)

for epoch in range(5000):
    order = np.random.permutation(len(x_train))

    for start_index in range(0, len(x_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index: start_index+batch_size]

        x_batch = x_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)

        preds = net.forward(x_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()