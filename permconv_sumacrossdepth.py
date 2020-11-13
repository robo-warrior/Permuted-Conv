import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import time


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 500
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print("Dataset size = ", len(trainloader))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.heightwidth = 32

        # permconv1
        self.in_channels1 = 3
        self.permfilters1 = 10
        self.outchannelsdepthwise1 = self.in_channels1 * self.permfilters1
        self.kernelsize1 = 3
        self.stride1 = 1
        self.padding1 = 1
        self.depthwise_conv1 = nn.Conv2d(self.in_channels1, self.outchannelsdepthwise1,
                                         kernel_size=self.kernelsize1, groups=self.in_channels1, stride=self.stride1,
                                         padding=self.padding1)
        self.num_perm1 = (int)(self.permfilters1 + self.in_channels1 * ((self.permfilters1 - 1) * (self.permfilters1)) / 2)
        self.outchannelsperm1 = 100
        self.oneXone_conv1 = nn.Conv2d(self.num_perm1, self.outchannelsperm1, kernel_size=1)
        self.heightwidthafter1 = (int)(((self.heightwidth - self.kernelsize1 + 2 * self.padding1)/self.stride1) + 1)

        # pool1
        self.maxpoolscale1 = 2
        self.pool1 = nn.MaxPool2d(self.maxpoolscale1, self.maxpoolscale1)
        self.heightwidthafterpool1 = (int)(self.heightwidthafter1/2)

        # permconv2
        self.in_channels2 = self.outchannelsperm1
        self.permfilters2 = 6
        self.outchannelsdepthwise2 = self.in_channels2 * self.permfilters2
        self.kernelsize2 = 3
        self.stride2 = 1
        self.padding2 = 1
        self.depthwise_conv2 = nn.Conv2d(self.in_channels2, self.outchannelsdepthwise2,
                                         kernel_size=self.kernelsize2, groups=self.in_channels2, stride=self.stride2,
                                         padding=self.padding2)
        self.num_perm2 = (int)(self.permfilters2 + self.in_channels2 * ((self.permfilters2 - 1) * (self.permfilters2)) / 2)
        self.outchannelsperm2 = 300
        self.oneXone_conv2 = nn.Conv2d(self.num_perm2, self.outchannelsperm2, kernel_size=1)
        self.heightwidthafter2 = (int)(((self.heightwidthafterpool1 - self.kernelsize2 + 2 * self.padding2) / self.stride2) + 1)

        # pool1
        self.maxpoolscale2 = 2
        self.pool2 = nn.MaxPool2d(self.maxpoolscale2, self.maxpoolscale2)
        self.heightwidthafterpool2 = (int)(self.heightwidthafter2 / 2)

        # fc1
        self.outchannelsfc1 = 120
        self.fc1 = nn.Linear(self.heightwidthafterpool2 * self.heightwidthafterpool2 * self.outchannelsperm2, self.outchannelsfc1)

        # fc2
        self.outchannelsfc2 = 10
        self.fc2 = nn.Linear(self.outchannelsfc1, self.outchannelsfc2)

        self.global_permconv_count = 0


    # perform depthwise conv, perform typical summation across permuted channels (no 1x1 conv), perform 1x1 convolution to reduce channels
    def permconv(self, x, out_channels, num_filters, kernel_size, stride, padding):
        x = x.to(device)
        self.global_permconv_count += 1

        # # depthwise conv
        # self.add_module('depthwise_conv_%d' % self.global_permconv_count,
        #                 nn.Conv2d(in_channels, in_channels * num_filters, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding))

        x = getattr(self, 'depthwise_conv%d' % self.global_permconv_count)(x)
        x = x.to(device)

        # permuted summation
        num_channels_per_filter = int(x.shape[1] / num_filters)
        # print("num_channels_per_filter", num_channels_per_filter)
        # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
        total_groups_of_channels = (int)(num_filters + num_channels_per_filter * ((num_filters - 1) * (num_filters)) / 2)
        x_out = torch.zeros((x.shape[0], total_groups_of_channels, x.shape[2], x.shape[3]))
        x_out = x_out.to(device)

        count = 0
        for i in range(num_filters):
            for j in range(num_channels_per_filter):
                if (j == 0):
                    for k in range(i, num_filters):
                        x_out[:, count, :, :] += x[:, i * num_channels_per_filter + j, :, :]
                        x_out[:, count, :, :] += torch.sum(x[:,k * num_channels_per_filter + 1: k * num_channels_per_filter + num_channels_per_filter,
                                                                                    :, :], dim=1)
                        count += 1
                else:
                    for k in range(i + 1, num_filters):
                        x_out[:, count, :, :] += x[:, i * num_channels_per_filter + j, :, :]
                        # print("Count", count, "count + j", count + j, "k * num_channels_per_filter", k * num_channels_per_filter, "k * num_channels_per_filter + j", k * num_channels_per_filter + j)
                        x_out[:, count, :, :] += torch.sum(x[:, k * num_channels_per_filter: k * num_channels_per_filter + j, :, :], dim=1)
                        x_out[:, count, :, :] += torch.sum(x[:,k * num_channels_per_filter + j + 1: k * num_channels_per_filter + num_channels_per_filter,
                                                                                        :, :], dim=1)
                        count += 1

        x_out = F.relu(x_out)
        # 1x1 conv
        # self.add_module('oneXone_conv%d' % self.global_permconv_count,
        #                 nn.Conv2d(total_groups_of_channels, out_channels, kernel_size=1))

        x = getattr(self, 'oneXone_conv%d' % self.global_permconv_count)(x_out)
        x = x.to(device)

        return x


    def forward(self, x):
        print("num_perm1", self.num_perm1)
        x = self.permconv(x, out_channels=self.outchannelsperm1, num_filters=self.permfilters1, kernel_size=self.kernelsize1, stride=self.stride1, padding=self.padding1)
        x = F.relu(x)
        x = self.pool1(x)
        print("num_perm2", self.num_perm2)
        x = self.permconv(x, out_channels=self.outchannelsperm2, num_filters=self.permfilters2, kernel_size=self.kernelsize2, stride=self.stride2, padding=self.padding2)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        self.global_permconv_count = 0
        print("Forward Done")
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        start_time = time.time()
        print("Epoch", epoch, "iteration", i)
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

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i, running_loss / 2000))
            running_loss = 0.0
        end_time = time.time()
        print("Time", end_time-start_time)

    if(epoch%2 == 0):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on test images: %0.3f %%' % (
                100 * correct / total))

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))