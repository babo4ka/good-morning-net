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


def get_avg(x):
    sumT = 0
    for i in x:
        sumT += i
    return sumT / len(x)


for line in file:
    input_strings.append(line.rstrip())

x_data = {}
avg_list = []

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

    avg = get_avg(hoursArr)

    avg_list.append(avg)

avg_list = np.array(avg_list)  # .reshape(-1, 1)

test_size = 15
avg_train = avg_list[:-test_size]
avg_test = avg_list[-test_size:]


max_avg = max(avg_train)

avg_train = [x / max_avg for x in avg_train]

avg_train = torch.FloatTensor(avg_train).unsqueeze(1).view(-1)

train_window = 7


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


avg_train_io_seq = create_inout_sequences(avg_train, train_window)


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


avg_net = GoodMoTimeNet()
loss_fc = nn.MSELoss()
optimizer = optim.Adam(avg_net.parameters(), lr=1.0e-3)

for i in range(2000):
    for seq, labels in avg_train_io_seq:
        optimizer.zero_grad()
        avg_net.hidden_cell = (torch.zeros(1, 1, avg_net.hidden_layer_size),
                               torch.zeros(1, 1, avg_net.hidden_layer_size))

        y_pred = avg_net(seq)

        single_loss = loss_fc(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 100 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 15

test_inputs = avg_train[-train_window:].tolist()

avg_net.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])

    with torch.no_grad():
        avg_net.hidden = (torch.zeros(1, 1, avg_net.hidden_layer_size),
                          torch.zeros(1, 1, avg_net.hidden_layer_size))
    test_inputs.append(avg_net(seq).item())

actual_preds = [x * max_avg for x in test_inputs[train_window:]]

x = np.arange(28, 43, 1)

plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(avg_list, c='green')
plt.plot(x, actual_preds, c='red')
plt.show()




# rad_net = GoodMoTimeNet()
# loss_fc_rad = nn.MSELoss()
# optimizer_rad = optim.Adam(rad_net.parameters(), lr=1.0e-3)
#
# for i in range(400):
#     for seq, labels in radius_train_io_seq:
#         optimizer_rad.zero_grad()
#         rad_net.hidden_cell = (torch.zeros(1, 1, rad_net.hidden_layer_size),
#                                torch.zeros(1, 1, rad_net.hidden_layer_size))
#
#         y_pred = rad_net(seq)
#
#         single_loss = loss_fc_rad(y_pred, labels)
#         single_loss.backward()
#         optimizer_rad.step()
#
#     if i % 25 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
#
#         print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
#
# test_inputs_rad = radius_train[-train_window:].tolist()
#
# rad_net.eval()
#
# for i in range(fut_pred):
#     seq = torch.FloatTensor(test_inputs_rad[-train_window:])
#
#     with torch.no_grad():
#         rad_net.hidden = (torch.zeros(1, 1, rad_net.hidden_layer_size),
#                           torch.zeros(1, 1, rad_net.hidden_layer_size))
#     test_inputs_rad.append(rad_net(seq).item())
#
# actual_preds_rad = [x * max_radius for x in test_inputs_rad[train_window:]]
#
#
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(radius_list, c='green')
# plt.plot(x, actual_preds_rad, c='red')
# plt.show()
