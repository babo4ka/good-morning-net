from datetime import datetime
import torch
import torch.nn as nn

def get_avg(x):
    sumT = 0
    for i in x:
        sumT += i
    return sumT / len(x)


def get_rad(x):
    sumS = 0
    for i in range(len(x)):
        if i < len(x) - 1:
            if x[i + 1] > x[i]:
                sumS += x[i + 1] - x[i]
            else:
                sumS += x[i] - x[i + 1]

    return sumS / (len(x) - 1)


file = open('../resources/time_interval_nets/dates.txt', 'r')

full_date_format = '%d-%m-%Y %H:%M:%S'
date_format = '%d-%m-%Y'


def dates_to_dict():
    input_strings = list()
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
    return x_data


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class GoodMoTimeNet(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer_size)
        # self.linear2 = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        # lstm_out, self.hidden_cell = self.lstm2(predictions.view(len(predictions), 1, -1), self.hidden_cell)
        # predictions = self.linear2(lstm_out.view(len(predictions), -1))

        return predictions[-1]