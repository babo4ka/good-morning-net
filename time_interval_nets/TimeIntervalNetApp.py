from datetime import datetime

import numpy as np
import torch

from TimeUntervalNetUtils import dates_to_dict
from TimeUntervalNetUtils import full_date_format
from TimeUntervalNetUtils import get_avg
from TimeUntervalNetUtils import get_rad

import socket
import json

avg_net = torch.load('../resources/time_interval_nets/avg_net.pt')
avg_net.eval()
rad_net = torch.load('../resources/time_interval_nets/rad_net.pt')
rad_net.eval()

x_data = dates_to_dict()

avg_list = []
rad_list = []
for key in x_data.keys():
    hoursArr = list()
    for i in range(len(x_data[key])):
        time_obj = datetime.strptime(x_data[key][i], full_date_format)
        hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
        hoursArr.append(hours)

    avg = get_avg(hoursArr)
    rad = get_rad(hoursArr)

    avg_list.append(avg)
    rad_list.append(rad)

avg_list = np.array(avg_list)
rad_list = np.array(rad_list)

test_size = 16
train_window = 7

max_avg = max(avg_list[:-test_size])
max_rad = max(rad_list[:-test_size])

avg_test = avg_list[-test_size:]
rad_test = rad_list[-test_size:]

avg_inputs = avg_test[-train_window:].tolist()
rad_inputs = rad_test[-train_window:].tolist()


def predict():
    avg_seq = torch.FloatTensor(avg_inputs[-train_window:])
    rad_seq = torch.FloatTensor(rad_inputs[-train_window:])

    with torch.no_grad():
        avg_net.hidden = (torch.zeros(1, 1, avg_net.hidden_layer_size),
                          torch.zeros(1, 1, avg_net.hidden_layer_size))

        rad_net.hidden = (torch.zeros(1, 1, rad_net.hidden_layer_size),
                          torch.zeros(1, 1, rad_net.hidden_layer_size))

    avg = avg_net(avg_seq).item()
    rad = rad_net(rad_seq).item()

    avg_inputs.append(avg)
    rad_inputs.append(rad)

    return round((avg * max_avg - rad * max_rad), 2), round((avg * max_avg + rad * max_rad), 2)


HOST = ('192.168.0.103', 9999)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(HOST)
server.listen()
print("work")
while True:
    conn, addr = server.accept()
    data = conn.recv(4096)

    if data:
        start, end = predict()

        response = json.dumps({"start": start, "end":end})
        conn.send(response.encode())

    conn.close()