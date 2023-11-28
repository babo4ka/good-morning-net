import torch
from datetime import datetime

class MorningNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MorningNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class BlackBox():
    def __init__(self):
        self.avgNet = MorningNet(50)
        self.radiusNet = MorningNet(50)
        self.hours_train = list()
        self.avg_train = list()
        self.radius_train = list()
        self.input_strings = list()
        self.file = open('../resources/time_interval_nets/dates.txt', 'r')
        self.full_date_format = '%d-%m-%Y %H:%M:%S'
        self.date_format = '%d-%m-%Y'
        self.optimizerAvg = torch.optim.Adam(self.avgNet.parameters(), lr=0.01)
        self.optimizerRadius = torch.optim.Adam(self.radiusNet.parameters(), lr=0.01)

    def get_rad(self, x):
        sumS = 0
        for i in range(len(x)):
            if i < len(x) - 1:
                if x[i + 1] > x[i]:
                    sumS += x[i + 1] - x[i]
                else:
                    sumS += x[i] - x[i + 1]

        return sumS / (len(x) - 1)

    def get_avg(self, x):
        sumT = 0
        for i in x:
            sumT += i
        return sumT / len(x)

    def load_data(self):
        input_strings = list()
        file = open('../resources/time_interval_nets/dates.txt', 'r')

        for line in file:
            input_strings.append(line.rstrip())

        x_data = {}

        for str in input_strings:
            time_obj = datetime.strptime(str, self.full_date_format)
            if datetime.strftime(time_obj.date(), self.date_format) in x_data:
                x_data[datetime.strftime(time_obj.date(), self.date_format)].append(
                    datetime.strftime(time_obj, self.full_date_format))
            else:
                x_data[datetime.strftime(time_obj.date(), self.date_format)] = list()
                x_data[datetime.strftime(time_obj.date(), self.date_format)].append(
                    datetime.strftime(time_obj, self.full_date_format))

        for key in x_data.keys():
            hoursArr = list()
            for i in range(len(x_data[key])):
                time_obj = datetime.strptime(x_data[key][i], self.full_date_format)
                hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
                hoursArr.append(hours)
                self.hours_train.append(hours)

            avg = self.get_avg(hoursArr)
            rad = self.get_rad(hoursArr)
            for i in range(len(x_data[key])):
                self.avg_train.append(avg)
                self.radius_train.append(rad)

        self.hours_train = torch.tensor(self.hours_train)

        self.avg_train = torch.tensor(self.avg_train)
        self.radius_train = torch.tensor(self.radius_train)

        noise_avg = torch.randn(self.avg_train.size()) / 5
        noise_radius = torch.randn(self.radius_train.size()) / 5

        self.avg_train = self.avg_train + noise_avg
        self.radius_train = self.radius_train + noise_radius

        self.hours_train.unsqueeze_(1)
        self.avg_train.unsqueeze_(1)
        self.radius_train.unsqueeze_(1)

    def loss(self, pred, target):
        squares = (pred - target) ** 2
        return squares.mean()

    def learnBox(self):
        for epoch_index in range(2000):
            self.optimizerAvg.zero_grad()

            y_pred = self.avgNet.forward(self.hours_train)
            loss_val = self.loss(y_pred, self.avg_train)

            loss_val.backward()

            self.optimizerAvg.step()

        for epoch_index in range(2000):
            self.optimizerRadius.zero_grad()

            y_pred = self.radiusNet.forward(self.hours_train)
            loss_val = self.loss(y_pred, self.radius_train)

            loss_val.backward()

            self.optimizerRadius.step()

    def getAvg(self, x):
        x = torch.tensor([x])
        return round(self.avgNet.forward(x).item(), 2)

    def getRad(self, x):
        x = torch.tensor([x])
        return round(self.radiusNet.forward(x).item(), 2)

    def getInterval(self, x):
        avg = self.getAvg(x)
        rad = self.getRad(x)
        return avg-rad, avg+rad
