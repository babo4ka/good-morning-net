from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from TimeUntervalNetUtils import GoodMoTimeNet
from TimeUntervalNetUtils import create_inout_sequences
from TimeUntervalNetUtils import dates_to_dict
from TimeUntervalNetUtils import full_date_format
from TimeUntervalNetUtils import get_avg

x_data = dates_to_dict()
avg_list = []


for key in x_data.keys():
    hoursArr = list()
    for i in range(len(x_data[key])):
        time_obj = datetime.strptime(x_data[key][i], full_date_format)
        hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
        hoursArr.append(hours)

    avg = get_avg(hoursArr)

    avg_list.append(avg)

avg_list = np.array(avg_list)

test_size = 16
avg_train = avg_list[:-test_size]
avg_test = avg_list[-test_size:]


max_avg = max(avg_train)

avg_train = [x / max_avg for x in avg_train]

avg_train = torch.FloatTensor(avg_train).unsqueeze(1).view(-1)

train_window = 7


avg_train_io_seq = create_inout_sequences(avg_train, train_window)


avg_net = GoodMoTimeNet(hidden_layer_size=250)
loss_fc = nn.MSELoss()
optimizer = optim.Adam(avg_net.parameters(), lr=1.0e-3)

for i in range(1700):
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

torch.save(avg_net, '../resources/time_interval_nets/avg_net.pt')
fut_pred = 7

test_inputs = avg_train[-train_window:].tolist()

avg_net.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])

    with torch.no_grad():
        avg_net.hidden = (torch.zeros(1, 1, avg_net.hidden_layer_size),
                          torch.zeros(1, 1, avg_net.hidden_layer_size))
    test_inputs.append(avg_net(seq).item())

actual_preds = [x * max_avg for x in test_inputs[train_window:]]

x = np.arange(37, 44, 1)

plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(avg_list, c='green')
plt.plot(x, actual_preds, c='red')
plt.show()

