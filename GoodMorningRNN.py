import torch
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

input_strings = list()
file = open('./resources/dates.txt', 'r')

full_date_format = '%d-%m-%Y %H:%M:%S'
date_format = '%d-%m-%Y'

hours_train = list()
avg_train = list()
radius_train = list()


def get_rad(x):
    sumS = 0
    for i in range(len(x)):
        if i < len(x) - 1:
            if x[i + 1] > x[i]:
                sumS += x[i + 1] - x[i]
            else:
                sumS += x[i] - x[i + 1]

    return sumS / (len(x) - 1)


def get_avg(x):
    sumT = 0
    for i in x:
        sumT += i
    return sumT / len(x)


for line in file:
    input_strings.append(line.rstrip())

x_data = {}

for str in input_strings:
    time_obj = datetime.strptime(str, full_date_format)
    if datetime.strftime(time_obj.date(), date_format) in x_data:
        x_data[datetime.strftime(time_obj.date(), date_format)].append(
            datetime.strftime(time_obj, full_date_format))
    else:
        x_data[datetime.strftime(time_obj.date(), date_format)] = list()
        x_data[datetime.strftime(time_obj.date(), date_format)].append(
            datetime.strftime(time_obj, full_date_format))

for key in x_data.keys():
    hoursArr = list()
    for i in range(len(x_data[key])):
        time_obj = datetime.strptime(x_data[key][i], full_date_format)
        hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
        hoursArr.append(hours)
        hours_train.append(hours)

    avg = get_avg(hoursArr)
    rad = get_rad(hoursArr)
    for i in range(len(x_data[key])):
        avg_train.append(avg)
        radius_train.append(rad)


max_hours = max(hours_train)
max_avg = max(avg_train)
max_radius = max(radius_train)

hours_train = [x / max_hours for x in hours_train]
avg_train = [x / max_avg for x in avg_train]
radius_train = [x / max_radius for x in radius_train]

hours_train = np.array(hours_train)
avg_train = np.array(avg_train)
radius_train = np.array(radius_train)


hours_train = torch.tensor(hours_train, dtype=torch.float32)
avg_train = torch.tensor(avg_train, dtype=torch.float32)
radius_train = torch.tensor(radius_train, dtype=torch.float32)


plt.plot(hours_train, c='blue')
plt.plot(avg_train - radius_train, c='red')
plt.plot(avg_train + radius_train, c='green')
plt.show()





class GMNet(nn.Module):
    def __init__(self, in_size, hidden_neurons, layers):
        super(GMNet, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_neurons, layers, batch_first=True)
        self.fc = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

hours_train = hours_train.reshape([-1, 1]).to(device)
avg_train = avg_train.reshape([-1, 1]).to(device)
radius_train = radius_train.reshape([-1, 1]).to(device)


print(hours_train.shape)
print(avg_train.shape)
print(radius_train.shape)


net = GMNet(1, 64, 2).to(device)

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.Adam(net.parameters(), lr=1e-3)

predictions = []

for epoch in range(50):
    net.train()
    i = 0
    for x in hours_train:
        print(x)
        preds = net(x)

        predictions.append(preds)
        loss = loss_fn(preds, avg_train[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1


plt.plot(predictions, c='red')
plt.plot(avg_train.cpu(), c='green')
plt.show()