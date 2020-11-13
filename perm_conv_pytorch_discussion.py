import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batchsize = 500

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batchsize,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5) # (10,28,28)
        self.conv2 = nn.Conv2d(10, 20, 5) # (20,24,24)

        self.pool = nn.MaxPool2d(2, 2) # (20,12,12)

        self.conv3 = nn.Conv2d(20, 29, 3) # (29,10,10)

        # myconv1*1 = (14,10,10) [1x1conv on channels: (0,1,2), (0,3,4), (0,5,6), (0,7,8)]

        self.fc1 = nn.Linear(14*10*10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # x = 60 (20*3) --> 0,1,...59  -> 0,4,7 -> 1*1 -> 1 output
    def myconv1x1(self, x): # (9,10,10)
        x_in = torch.zeros((x.shape[0], 3, x.shape[2], x.shape[3])) # (3,10,10)
        x_out = torch.zeros(x.shape[0], int((x.shape[1]-1)/2), x.shape[2], x.shape[3]) # (4,10,10)
        x_out = x_out.to(device)
        for i in range(x_out.shape[1]): # 4
            x_in[:, 0, :, :] = x[:, 0, :, :]
            x_in[:, 1, :, :] = x[:, (i*2)+1, :, :]
            x_in[:, 2, :, :] = x[:, (i*2)+2, :, :]
            x_out[:, i, :, :] = torch.squeeze(nn.Conv2d(3, 1, kernel_size=1)(x_in))
        return x_out


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.myconv1x1(x)
        x = x.view(-1, 14*10*10)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(500):  # loop over the dataset multiple times
    # running_loss = 0.0
    print("Epoch", epoch)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        torch.autograd.set_detect_anomaly(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 0:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch, i, running_loss / 2000))
        #     running_loss = 0.0
