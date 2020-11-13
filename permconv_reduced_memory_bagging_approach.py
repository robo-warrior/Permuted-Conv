from comet_ml import Experiment
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
import torch.optim.lr_scheduler as lrs

experiment = Experiment(api_key="hPc2DeBWvLYMqWUFLMgVTSQrF",
                        project_name="permuted-convolutions", workspace="rishabh")

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

print("Number of iterations per epoch = ", len(trainloader))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.heightwidth = 32

        # depthwise separable 1
        self.n_in1 = 3
        self.kernels_layer1 = 25
        self.filter_size1 = 3
        self.channelsafter1 = self.kernels_layer1
        self.depthwise1 = nn.Conv2d(self.n_in1, self.n_in1 * self.kernels_layer1, kernel_size=self.filter_size1, groups=self.n_in1)
        self.oneXoneDepthSeparable1 = nn.Conv2d(self.n_in1 * self.kernels_layer1, self.channelsafter1, kernel_size=1, groups=self.kernels_layer1)
        self.heightwidthafter1 = self.heightwidth - self.filter_size1 + 1

        # pool1
        self.maxpoolscale1 = 2
        self.pool1 = nn.MaxPool2d(self.maxpoolscale1, self.maxpoolscale1)
        self.heightwidthafterpool1 = (int)(self.heightwidthafter1/2)

        # depthwise separable 2
        self.n_in2 = self.channelsafter1
        self.kernels_layer2 = 40
        self.filter_size2 = 3
        self.channelsafter2 = self.kernels_layer2
        self.depthwise2 = nn.Conv2d(self.channelsafter1, self.n_in2 * self.kernels_layer2, kernel_size=self.filter_size2, groups=self.n_in2)
        self.oneXoneDepthSeparable2 = nn.Conv2d(self.n_in2 * self.kernels_layer2, self.channelsafter2, kernel_size=1, groups=self.kernels_layer2)
        self.heightwidthafter2 = self.heightwidthafterpool1 - self.filter_size2 + 1

        # pool2
        self.maxpoolscale2 = 2
        self.pool2 = nn.MaxPool2d(self.maxpoolscale2, self.maxpoolscale2)
        self.heightwidthafterpool2 = (int)(self.heightwidthafter2 / 2)

        # depthwise separable 3
        self.n_in3 = self.channelsafter2
        self.kernels_layer3 = 50
        self.filter_size3 = 3
        self.channelsafter3 = self.kernels_layer3
        self.depthwise3 = nn.Conv2d(self.channelsafter2, self.n_in3 * self.kernels_layer3, kernel_size=self.filter_size3,
                                    groups=self.n_in3)
        self.oneXoneDepthSeparable3 = nn.Conv2d(self.n_in3 * self.kernels_layer3, self.channelsafter3, kernel_size=1,  groups=self.kernels_layer3)
        self.heightwidthafter3 = self.heightwidthafterpool2 - self.filter_size3 + 1

        # fc1
        self.outchannelsfc1 = 120
        self.fc1 = nn.Linear(self.heightwidthafter3 * self.heightwidthafter3 * self.channelsafter3, self.outchannelsfc1)

        # fc2
        self.outchannelsfc2 = 10
        self.fc2 = nn.Linear(self.outchannelsfc1, self.outchannelsfc2)

    def permute(self, x, n_in, kernels_layer, num_channels_to_permute):
        x_new = torch.clone(x)
        for i in range(num_channels_to_permute):
            idx = torch.randperm(kernels_layer)
            idx = (idx * n_in) + i
            x_new[:, i:x.shape[1]:n_in, :, :] = x[:, idx, :, :]
        return x_new


    def forward(self, x):
        x = self.depthwise1(x)
        x = self.oneXoneDepthSeparable1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.depthwise2(x)
        # permute all channels
        x = self.permute(x, self.n_in2, self.kernels_layer2, 25)
        x = self.oneXoneDepthSeparable2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.depthwise3(x)
        # permute all channels
        # x = self.permute(x, self.n_in3, self.kernels_layer3, 5)
        x = self.oneXoneDepthSeparable3(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
model_scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
training_acc_list = []
testing_acc_list = []
training_loss_list = []
testing_loss_list = []
loss_per_epoch = []
test_every = 5

for epoch in range(10001):  # loop over the dataset multiple times
    running_loss = 0.0
    epoch_loss = []
    net.train()
    for i, data in enumerate(trainloader, 0):
        start_time = time.time()
        # print("Epoch", epoch, "iteration", i)
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        end_time = time.time()
        # print("Time", end_time-start_time)
    loss_per_epoch.append(np.mean(epoch_loss))
    model_scheduler.step(np.mean(epoch_loss))
    plt.plot(loss_per_epoch, color='red', label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss plot per epoch')
    plt.legend()
    plt.savefig("./loss_per_epoch_plot_perm2.png", format='png')
    experiment.log_figure(figure=plt, figure_name='loss_per_epoch_plot_perm', overwrite=True)
    plt.close()

    if(epoch%5 == 0):

        # test
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Epoch:", epoch, 'Accuracy of the network on test images: %0.3f %%' % (
                100 * correct / total))
        testing_acc_list.append(100 * correct / total)
        testing_loss_list.append(test_loss/len(testloader))

        # train
        correct = 0
        total = 0
        train_loss = 0
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Epoch:", epoch, 'Accuracy of the network on training images: %0.3f %%' % (
                100 * correct / total))
        training_acc_list.append(100 * correct / total)
        training_loss_list.append(train_loss/len(trainloader))

        plt.plot(training_loss_list, color='blue', label='Training')
        plt.plot(testing_loss_list, color='red', label='Testing', alpha=.5)
        plt.xlabel('Epoch/{}'.format(test_every))
        plt.ylabel('Loss')
        plt.title('Loss plot')
        plt.legend()
        plt.savefig("./loss_plot_perm2.png", format='png')
        experiment.log_figure(figure=plt, figure_name='loss_plot_perm', overwrite=True)
        plt.close()

        plt.plot(training_acc_list, color='blue', label='Training')
        plt.plot(testing_acc_list, color='red', label='Testing', alpha=.5)
        plt.xlabel('Epoch/{}'.format(test_every))
        plt.ylabel('Accuracy')
        plt.title('Accuracy plot')
        plt.legend()
        plt.savefig("./accuracy_plot_perm2.png", format='png')
        experiment.log_figure(figure=plt,  figure_name='accuracy_plot_perm', overwrite=True)
        plt.close()


print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

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