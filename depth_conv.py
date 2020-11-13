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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=500,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=500,
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

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print("Dataset size = ", len(trainloader))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        nin = 3
        kernels_per_layer = 6
        self.conv1 = nn.Conv2d(3, 6, 5)

        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, groups=nin)
        self.oneXoneDepthSeparable = nn.Conv2d(18, 21, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)



        self.test1x1 = nn.Conv2d(3, 1, kernel_size=1)



        nin2 = int((kernels_per_layer*(kernels_per_layer+1))/2)
        kernels_per_layer2 = 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.depthwise2 = nn.Conv2d(nin2, nin2 * kernels_per_layer2, kernel_size=5, groups=nin2)
        self.oneXoneDepthSeparable2 = nn.Conv2d(336, 136, kernel_size=1)

        self.fc1 = nn.Linear(136*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def permuted_oneXone_conv(self, x, nin):
        num_filters = x.shape[1]/nin
        print("Num filters", int(num_filters))
        x_in = torch.zeros((x.shape[0], nin, x.shape[2], x.shape[3]))
        # print("X in", x_in.shape)

        # permuting first channel
        x_out = torch.zeros(x.shape[0], int((num_filters * (num_filters+1))/2), x.shape[2], x.shape[3])
        x_out = x_out.to(device)
        count = 0
        #(x) 0,1,...59 --> 0,1,2 3,1,2 6,1,2 .... (3,4,5)  --> 3,4,5 6,4,5 9,4,5...
        for i in range(int(num_filters)):
            # print("*********" + str(i) + "***********")
            for j in range(i, int(num_filters)):
                x_in[:, 0, :, :] = x[:, j*nin, :, :]
                x_in[:, 1:, :, :] = x[:, (i*nin)+1:(i+1)*nin, :, :]
                x_out_channel = nn.Conv2d(nin, 1, kernel_size=1)(x_in)
                x_out[:, count, :, :] = x_out_channel
                # print(count)
                count = count + 1
        return x_out

    # def permuted_oneXone_conv(self, x, nin):
    #     num_filters = x.shape[1] / nin
    #     print("Num filters", int(num_filters))
    #     x_in = torch.zeros((x.shape[0], nin, x.shape[2], x.shape[3]))
    #     # print("X in", x_in.shape)
    #
    #     # permuting first channel
    #     x_out = torch.zeros(x.shape[0], int((num_filters * (num_filters + 1)) / 2), x.shape[2], x.shape[3])
    #     x_out = x_out.to(device)
    #     count = 0
    #     for i in range(int(num_filters)):
    #         # print("*********" + str(i) + "***********")
    #         for j in range(i, int(num_filters)):
    #             x_in[:, 0, :, :] = x[:, j * nin, :, :]
    #             x_in[:, 1:, :, :] = x[:, (i * nin) + 1:(i + 1) * nin, :, :]
    #             x_out_channel = nn.Conv2d(nin, 1, kernel_size=1)(x_in)
    #             x_out[:, count, :, :] = x_out_channel
    #             # print(count)
    #             count = count + 1
    #     return x_out



    # def permuted_oneXone_conv(self, x, nin):
    #     num_filters = x.shape[1]/nin
    #     print("Num filters", int(num_filters))
    #     x_in = torch.zeros((x.shape[0], nin, x.shape[2], x.shape[3]))
    #     print("X in", x_in.shape)
    #
    #     # permuting first channel
    #     x_out = torch.zeros(x.shape[0], int((num_filters * (num_filters+1))/2), x.shape[2], x.shape[3])
    #     x_out_list = []
    #
    #     x_in = x_in.to(device)
    #     x_out_tens = torch.zeros(1, 1, 28, 28)
    #     # x_out_tens = x_out_tens.to(device)
    #
    #     # x_out = x_out.to(device)
    #     count = 0
    #
    #     for i in range(int(num_filters)):
    #         # print("*********" + str(i) + "***********")
    #         for j in range(i, int(num_filters)):
    #             x_in[:, 0, :, :] = x[:, j*nin, :, :]
    #             x_in[:, 1:, :, :] = x[:, (i*nin)+1:(i+1)*nin, :, :]
    #             x_in = x_in.detach().cpu()
    #             temp = nn.Conv2d(3, 1, kernel_size=1)(x_in)
    #
    #             print("Temp", temp.shape)
    #             x_out_tens = torch.cat((x_out_tens, temp), 1)
    #             print("X out tens", x_out_tens.shape)
    #             # x_out_list.append(temp)
    #             # x_out[:, count, :, :] = nn.Conv2d(3, 1, kernel_size=1)(x_in)
    #             # print(count)
    #             count = count + 1
    #     # torch.cat(x_out_list, out=x_out)
    #     x_out_tens = x_out_tens[:, 1:, :, :]
    #     x_out_tens.to(device)
    #     return x_out_tens


    # n = num_filters, n_in = no of input channels, n_out = no of output channels
    # unique combinations across filters => n_out = n^2 + n_in * (n-1)^2
    # (n-1)^3 < n^2 + n_in * (n-1)^2 < n^3

    # 5 filters, n_in = 3
    # perm 1st =     --> (123, 423, 723, 10-2-3, 13-2-3), (156, 456, 756,..), .... --> 25
    # perm 2nd =     --> (153, 183, 1-11-3, )

    def forward(self, x):
        # x = self.conv1(x)
        nin = x.shape[1]
        # print("Before 1", x.shape)
        depthwise_x = self.depthwise(x)
        # print("Depthwise 1", depthwise_x.shape)
        # x = self.permuted_oneXone_conv(depthwise_x, nin)
        # x = x.to(device)

        x = self.oneXoneDepthSeparable(depthwise_x)

        # print("After 1", x.shape)
        # x = self.oneXone(x)
        # print("oneXone", x.shape)
        x = self.pool(F.relu(x))
        # x = self.conv2(x)
        nin = x.shape[1]
        # print("Before 2", x.shape)
        depthwise_x = self.depthwise2(x)
        # print("Depthwise 2", depthwise_x.shape)

        # x = self.permuted_oneXone_conv(depthwise_x, nin)

        x = self.oneXoneDepthSeparable2(depthwise_x)

        # print("After 2", x.shape)
        x = self.pool(F.relu(x))

        x = x.view(-1, 136*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
    running_loss = 0.0
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

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i, running_loss / 2000))
            running_loss = 0.0

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