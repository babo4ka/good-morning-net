import torch
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from TimeUntervalNetUtils import get_rad
from TimeUntervalNetUtils import dates_to_dict
from TimeUntervalNetUtils import full_date_format
from TimeUntervalNetUtils import create_inout_sequences
from TimeUntervalNetUtils import GoodMoTimeNet

x_data = dates_to_dict()
radius_list = []

for key in x_data.keys():
    hoursArr = list()
    for i in range(len(x_data[key])):
        time_obj = datetime.strptime(x_data[key][i], full_date_format)
        hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
        hoursArr.append(hours)

    rad = get_rad(hoursArr)

    radius_list.append(rad)

radius_list = np.array(radius_list)

test_size = 16

radius_train = radius_list[:-test_size]
radius_test = radius_list[-test_size:]

max_radius = max(radius_train)

radius_train = [x / max_radius for x in radius_train]

radius_train = torch.FloatTensor(radius_train).unsqueeze(1).view(-1)

train_window = 7

radius_train_io_seq = create_inout_sequences(radius_train, train_window)

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

# torch.save(rad_net, '../resources/time_interval_nets/rad_net.pt')

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
