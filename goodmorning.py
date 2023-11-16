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
        self.startNet = MorningNet(50)
        self.endNet = MorningNet(50)
        self.x_train = list()
        self.radiuses = list()
        self.y_start_train = None
        self.y_end_train = None
        self.input_strings = list()
        self.file = open('./resources/dates.txt', 'r')
        self.full_date_format = '%d-%m-%Y %H:%M:%S'
        self.date_format = '%d-%m-%Y'
        self.optimizerStart = torch.optim.Adam(self.startNet.parameters(), lr=0.01)
        self.optimizerEnd = torch.optim.Adam(self.endNet.parameters(), lr=0.01)

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
        file = open('./resources/dates.txt', 'r')

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
            sumT = 0
            sumS = 0
            hoursArr = list()
            for i in range(len(x_data[key])):
                time_obj = datetime.strptime(x_data[key][i], self.full_date_format)
                hours = int(time_obj.strftime('%H')) + int(time_obj.strftime('%M')) / 60
                sumT += hours
                hoursArr.append(hours)

                if i < len(x_data[key]) - 1:
                    nextTimeObj = datetime.strptime(x_data[key][i + 1], self.full_date_format)
                    nextHours = int(nextTimeObj.strftime('%H')) + int(nextTimeObj.strftime('%M')) / 60
                    sumS += nextHours - hours
            self.x_train.append(self.get_avg(hoursArr))
            self.radiuses.append(self.get_rad(hoursArr))
        self.x_train = torch.tensor(self.x_train)
        self.radiuses = torch.tensor(self.radiuses)

        self.y_start_train = self.x_train - self.radiuses
        self.y_end_train = self.x_train + self.radiuses

        noise_start = torch.randn(self.y_start_train.size()) / 5
        noise_end = torch.randn(self.y_end_train.size()) / 5

        self.y_start_train = self.y_start_train + noise_start
        self.y_end_train = self.y_end_train + noise_end

        self.x_train.unsqueeze_(1)
        self.y_start_train.unsqueeze_(1)
        self.y_end_train.unsqueeze_(1)

    def loss(self, pred, target):
        squares = (pred - target) ** 2
        return squares.mean()

    def learnBox(self):
        for epoch_index in range(2000):
            self.optimizerStart.zero_grad()

            y_pred = self.startNet.forward(self.x_train)
            loss_val = self.loss(y_pred, self.y_start_train)

            loss_val.backward()

            self.optimizerStart.step()

        for epoch_index in range(2000):
            self.optimizerEnd.zero_grad()

            y_pred = self.endNet.forward(self.x_train)
            loss_val = self.loss(y_pred, self.y_end_train)

            loss_val.backward()

            self.optimizerEnd.step()

    def getStart(self, x):
        x = torch.tensor([x])
        return self.startNet.forward(x)

    def getEnd(self, x):
        x = torch.tensor([x])
        return self.endNet.forward(x)

