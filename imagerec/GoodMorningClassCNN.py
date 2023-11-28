import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Dataset import MyDataset


class GMImagesNet(nn.Module):
    def __init__(self):
        super(GMImagesNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.act3 = nn.Tanh()

        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Tanh()

        self.fc3 = nn.Linear(84, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x



if __name__ == "__main__":
    dataset = MyDataset(file="../resources/img_resources/annotations.csv", root_dir='../resources/img_resources/imgs',
                        transform=transforms.ToTensor())

    train_set, test_set = torch.utils.data.random_split(dataset, [1000, 266])
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

    x_test = torch.stack(x).float()
    y_test = torch.tensor(y).float()

    net = GMImagesNet()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = 50

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(2000):
        order = np.random.permutation(len(x_train))

        for start_index in range(0, len(x_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index: start_index + batch_size]

            x_batch = x_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(x_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = net.forward(x_test)

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
        if accuracy > 0.95:
            break
        print(accuracy)

    torch.save(net, "../resources/img_resources/goodmrngnet.pt")


# path = './resources/imgs/mrng1.jpg'
# image = Image.open(path).convert('RGB')
# image = image.resize((28, 28))
# t = transforms.ToTensor()
# img_as_ten = t(image)
# x_inp = torch.stack([img_as_ten]).float()
# result = net.forward(x_inp)
# result = result.argmax(dim=1)
# print(result)
