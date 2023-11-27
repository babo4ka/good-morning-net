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


def get_rad(x):
    sumS = 0
    for i in range(len(x)):
        if i < len(x) - 1:
            if x[i + 1] > x[i]:
                sumS += x[i + 1] - x[i]
            else:
                sumS += x[i] - x[i + 1]

    return sumS / (len(x) - 1)



for line in file:
    input_strings.append(line.rstrip())

x_data = {}
radius_list = []

for str in input_strings:
    time_obj = datetime.strptime(str, full_date_format)
    if datetime.strftime(time_obj.date(), date_format) in x_data:
        x_data[datetime.strftime(time_obj.date(), date_format)].append(
            datetime.strftime(time_obj, full_date_format))
    else:
        x_data[datetime.strftime(time_obj.date(), date_format)] = list()
        x_data[datetime.strftime(time_obj.date(), date_format)].append(
            datetime.strftime(time_obj, full_date_format))

print(x_data)
for key in x_data.keys():
    hoursArr = list()
    for i in range(len(x_data[key])):
        time_obj = datetime.strptime(x_data[key][i], full_date_format)
        hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
        hoursArr.append(hours)

    rad = get_rad(hoursArr)

    radius_list.append(rad)

radius_list = np.array(radius_list)  # .reshape(-1, 1)

test_size = 15

radius_train = radius_list[:-test_size]
radius_test = radius_list[-test_size:]

max_radius = max(radius_train)

radius_train = [x / max_radius for x in radius_train]

radius_train = torch.FloatTensor(radius_train).unsqueeze(1).view(-1)

train_window = 7


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


radius_train_io_seq = create_inout_sequences(radius_train, train_window)


class GoodMoTimeNet(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


rad_net = GoodMoTimeNet()
loss_fc = nn.MSELoss()
optimizer = optim.Adam(rad_net.parameters(), lr=1.0e-3)

for i in range(2400):
    for seq, labels in radius_train_io_seq:
        optimizer.zero_grad()
        rad_net.hidden_cell = (torch.zeros(1, 1, rad_net.hidden_layer_size),
                               torch.zeros(1, 1, rad_net.hidden_layer_size))

        y_pred = rad_net(seq)

        single_loss = loss_fc(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 100 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 15

test_inputs = radius_train[-train_window:].tolist()

rad_net.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])

    with torch.no_grad():
        rad_net.hidden = (torch.zeros(1, 1, rad_net.hidden_layer_size),
                          torch.zeros(1, 1, rad_net.hidden_layer_size))
    test_inputs.append(rad_net(seq).item())

actual_preds = [x * max_radius for x in test_inputs[train_window:]]

x = np.arange(28, 43, 1)

plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(radius_list, c='green')
plt.plot(x, actual_preds, c='red')
plt.show()