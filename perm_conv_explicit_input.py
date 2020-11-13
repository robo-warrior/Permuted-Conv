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
        # self.conv1 = nn.Conv2d(3, 6, 5)

        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, groups=nin)
        # self.oneXoneDepthSeparable = nn.Conv2d(51, 21, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

        nin2 = 51
        # nin2 = 6
        # nin2 = 16
        # kernels_per_layer2 = 6
        kernels_per_layer2 = 5
        self.depthwise2 = nn.Conv2d(nin2, nin2 * kernels_per_layer2, kernel_size=5, groups=nin2)


        self.permconv1 = nn.Conv2d(153, 51, kernel_size=1, groups=51)
        self.permconv2 = nn.Conv2d(26265, 515, kernel_size=1, groups=515)



        self.onexone = nn.Conv2d(515, 100, kernel_size=1)

        self.fc1 = nn.Linear(100*5*5, 120)



        # self.fc1 = nn.Linear(6 * 5 * 5, 120)
        # self.fc1 = nn.Linear(81 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    # def permuted_new(self, x, num_filters):
    #     num_channels_per_filter = int(x.shape[1]/num_filters)
    #     total_groups_of_channels = num_filters
    #     x_in = torch.zeros((x.shape[0], total_groups_of_channels * num_channels_per_filter, x.shape[2], x.shape[3]))
    #
    #
    #     # 1,2,3  1,5,6  1,8,9  ... 1,17,18   [1,2,3]  4,2,6  7,2,9  ... 16,2,18   [1,2,3]  4,5,3  7,8,3  ... 16,17,3
    #     #        4,5,6  4,7,8  ... 4,17,18           [4,5,6] 7,5,9  ... 16,5,18           [4,5,6] 7,8,6  ... 16,17,6
    #
    #     # n filters, 3 channels  --> (n + n-1 + n-1) + (n-1 + n-2 + n-2) + (n-2 + n-3 + n-3) + ... (n-(n-2) + n-(n-1) + n-(n-1)) + (n-(n-1) + n-n + n-n)
    #     # n + 3(n-1) + 3(n-2) + 3(n-3) + ... + 3(n-(n-1))
    #     # n + 3(n-1 + n-2 + ... + 2 + 1) = n + 3*(n-1)(n)/2
    #     # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
    #     for i in range(num_filters):
    #         x_in[:, i*num_channels_per_filter, :, :] = x[:, 0, :, :]
    #         x_in[:, (i*num_channels_per_filter)+1: (i+1)*num_channels_per_filter, :, :] = x[:, (i*num_channels_per_filter)+1 : (i+1)*num_channels_per_filter, :, :]
    #
    #     x_out = nn.Conv2d(x.shape[1], total_groups_of_channels, kernel_size=1, groups=total_groups_of_channels)(x_in)
    #     x_out = x_out.to(device)
    #     return x_out

    # def permuted_new_perfect(self, x, num_filters):
    #     x = x.to(device)
    #     num_channels_per_filter = int(x.shape[1]/num_filters)
    #     # print("num_channels_per_filter", num_channels_per_filter)
    #     # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
    #     total_groups_of_channels = num_filters + num_channels_per_filter * ((num_filters - 1)*(num_filters))/ 2
    #     print("total_groups_of_channels", total_groups_of_channels)
    #     total_groups_of_channels = int(total_groups_of_channels)
    #
    #     x_in = torch.zeros((x.shape[0], total_groups_of_channels * num_channels_per_filter, x.shape[2], x.shape[3]))
    #     x_in = x_in.to(device)
    #     print(x_in.shape)
    #
    #     # 1,2,3  1,5,6  1,8,9  ... 1,17,18   [1,2,3]  4,2,6  7,2,9  ... 16,2,18   [1,2,3]  4,5,3  7,8,3  ... 16,17,3
    #     #        4,5,6  4,7,8  ... 4,17,18           [4,5,6] 7,5,9  ... 16,5,18           [4,5,6] 7,8,6  ... 16,17,6
    #
    #     # n filters, 3 channels  --> (n + n-1 + n-1) + (n-1 + n-2 + n-2) + (n-2 + n-3 + n-3) + ... (n-(n-2) + n-(n-1) + n-(n-1)) + (n-(n-1) + n-n + n-n)
    #     # n + 3(n-1) + 3(n-2) + 3(n-3) + ... + 3(n-(n-1))
    #     # n + 3(n-1 + n-2 + ... + 2 + 1) = n + 3*(n-1)(n)/2
    #     # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
    #
    #     count = 0
    #     for i in range(num_filters):
    #         for j in range(num_channels_per_filter):
    #             if(j==0):
    #                 for k in range(i, num_filters):
    #                     x_in[:, count, :, :] = x[:, i * num_channels_per_filter + j, :, :]
    #                     x_in[:, count+1: count+num_channels_per_filter, :, :] = x[:, k * num_channels_per_filter + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
    #                     count += num_channels_per_filter
    #             else:
    #                 for k in range(i+1, num_filters):
    #                     x_in[:, count + j, :, :] = x[:, i * num_channels_per_filter + j, :, :]
    #                     # print("Count", count, "count + j", count + j, "k * num_channels_per_filter", k * num_channels_per_filter, "k * num_channels_per_filter + j", k * num_channels_per_filter + j)
    #                     x_in[:, count: count + j, :, :] = x[:, k * num_channels_per_filter : k * num_channels_per_filter + j, :, :]
    #                     x_in[:, count+j+1 : count + num_channels_per_filter, :, :] = x[:,k * num_channels_per_filter + j + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
    #                     count += num_channels_per_filter
    #             # print("Count", count)
    #
    #     # x_in = x_in.to(device)
    #     print(x_in.is_cuda)
    #     x_out = nn.Conv2d(x_in.shape[1], total_groups_of_channels, kernel_size=1, groups=total_groups_of_channels)(x_in)
    #     print(x_out.is_cuda)
    #     # x_out = x_out.to(device)
    #     return x_out

    def permuted_new_perfect(self, x, num_filters):
        x = x.to(device)
        num_channels_per_filter = int(x.shape[1]/num_filters)
        # print("num_channels_per_filter", num_channels_per_filter)
        # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
        total_groups_of_channels = num_filters + num_channels_per_filter * ((num_filters - 1)*(num_filters))/ 2
        print("total_groups_of_channels", total_groups_of_channels)
        total_groups_of_channels = int(total_groups_of_channels)

        x_in = torch.zeros((x.shape[0], total_groups_of_channels * num_channels_per_filter, x.shape[2], x.shape[3]))
        x_in = x_in.to(device)
        print(x_in.shape)


        # 1,2,3  1,5,6  1,8,9  ... 1,17,18   [1,2,3]  4,2,6  7,2,9  ... 16,2,18   [1,2,3]  4,5,3  7,8,3  ... 16,17,3
        #        4,5,6  4,7,8  ... 4,17,18           [4,5,6] 7,5,9  ... 16,5,18           [4,5,6] 7,8,6  ... 16,17,6

        # n filters, 3 channels  --> (n + n-1 + n-1) + (n-1 + n-2 + n-2) + (n-2 + n-3 + n-3) + ... (n-(n-2) + n-(n-1) + n-(n-1)) + (n-(n-1) + n-n + n-n)
        # n + 3(n-1) + 3(n-2) + 3(n-3) + ... + 3(n-(n-1))
        # n + 3(n-1 + n-2 + ... + 2 + 1) = n + 3*(n-1)(n)/2
        # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2

        count = 0
        for i in range(num_filters):
            for j in range(num_channels_per_filter):
                if(j==0):
                    for k in range(i, num_filters):
                        x_in[:, count, :, :] = x[:, i * num_channels_per_filter + j, :, :]
                        x_in[:, count+1: count+num_channels_per_filter, :, :] = x[:, k * num_channels_per_filter + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
                        count += num_channels_per_filter
                else:
                    for k in range(i+1, num_filters):
                        x_in[:, count + j, :, :] = x[:, i * num_channels_per_filter + j, :, :]
                        # print("Count", count, "count + j", count + j, "k * num_channels_per_filter", k * num_channels_per_filter, "k * num_channels_per_filter + j", k * num_channels_per_filter + j)
                        x_in[:, count: count + j, :, :] = x[:, k * num_channels_per_filter : k * num_channels_per_filter + j, :, :]
                        x_in[:, count+j+1 : count + num_channels_per_filter, :, :] = x[:, k * num_channels_per_filter + j + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
                        count += num_channels_per_filter
                # print("Count", count)

        # x_in = x_in.to(device)

        if(x_in.shape[1] == 153):
            x_out = self.permconv1(x_in)
        else:
            x_out = self.permconv2(x_in)
        return x_out

    # def permuted_new_one_filter(self, x, num_filters):
    #     num_channels_per_filter = int(x.shape[1]/num_filters)
    #     # print("num_channels_per_filter", num_channels_per_filter)
    #     # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
    #     # total_groups_of_channels = num_filters + num_channels_per_filter * ((num_filters - 1)*(num_filters))/ 2
    #     # print("total_groups_of_channels", total_groups_of_channels)
    #     # total_groups_of_channels = int(total_groups_of_channels)
    #
    #     total_groups_of_channels = num_channels_per_filter * num_filters - (num_channels_per_filter-1)
    #
    #     # x_in = torch.zeros((x.shape[0], total_groups_of_channels * num_channels_per_filter, x.shape[2], x.shape[3]))
    #     x_in = torch.zeros((x.shape[0], total_groups_of_channels * num_channels_per_filter, x.shape[2], x.shape[3]))
    #
    #     # 1,2,3  1,5,6  1,8,9  ... 1,17,18   [1,2,3]  4,2,6  7,2,9  ... 16,2,18   [1,2,3]  4,5,3  7,8,3  ... 16,17,3
    #     #        4,5,6  4,7,8  ... 4,17,18           [4,5,6] 7,5,9  ... 16,5,18           [4,5,6] 7,8,6  ... 16,17,6
    #
    #     # n filters, 3 channels  --> (n + n-1 + n-1) + (n-1 + n-2 + n-2) + (n-2 + n-3 + n-3) + ... (n-(n-2) + n-(n-1) + n-(n-1)) + (n-(n-1) + n-n + n-n)
    #     # n + 3(n-1) + 3(n-2) + 3(n-3) + ... + 3(n-(n-1))
    #     # n + 3(n-1 + n-2 + ... + 2 + 1) = n + 3*(n-1)(n)/2
    #     # In general for n filters each of k channels  -->  n + k*(n-1)(n)/2
    #
    #     count = 0
    #     # for i in range(num_filters):
    #     for i in range(1):
    #         for j in range(num_channels_per_filter):
    #             if(j==0):
    #                 for k in range(i, num_filters):
    #                     x_in[:, count, :, :] = x[:, i * num_channels_per_filter + j, :, :]
    #                     x_in[:, count+1: count+num_channels_per_filter, :, :] = x[:, k * num_channels_per_filter + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
    #                     count += num_channels_per_filter
    #             else:
    #                 for k in range(i+1, num_filters):
    #                     x_in[:, count + j, :, :] = x[:, i * num_channels_per_filter + j, :, :]
    #                     # print("Count", count, "count + j", count + j, "k * num_channels_per_filter", k * num_channels_per_filter, "k * num_channels_per_filter + j", k * num_channels_per_filter + j)
    #                     x_in[:, count: count + j, :, :] = x[:, k * num_channels_per_filter : k * num_channels_per_filter + j, :, :]
    #                     x_in[:, count+j+1 : count + num_channels_per_filter, :, :] = x[:,k * num_channels_per_filter + j + 1 : k * num_channels_per_filter + num_channels_per_filter, :, :]
    #                     count += num_channels_per_filter
    #             # print("Count", count)
    #         # x_in[:, i*3, :, :] = x[:, 0, :, :]
    #         # x_in[:, (i*3)+1: (i+1)*3, :, :] = x[:, (i*3)+1: (i+1)*3, :, :]
    #
    #     # print("total_groups_of_channels", total_groups_of_channels, "x_in", x_in.shape[1])
    #     x_out = nn.Conv2d(x_in.shape[1], total_groups_of_channels, kernel_size=1, groups=total_groups_of_channels)(x_in)
    #     x_out = x_out.to(device)
    #     return x_out


    def forward(self, x):
        # print("Before 1", x.shape)
        # 3 -> 18
        depthwise_x = self.depthwise(x)
        # print("After Depthwise 1", depthwise_x.shape)
        # x = F.relu(depthwise_x)

        # 18 -> 51
        x = self.permuted_new_perfect(depthwise_x, 6)
        # print("After Perm 1", x.shape)

        x = self.pool(F.relu(x))
        # print("After Pool 1", x.shape)

        # 51 -> 255
        depthwise2_x = self.depthwise2(x)
        # print("After Depthwise 2", depthwise2_x.shape)

        # 255 -> 515
        x = self.permuted_new_perfect(depthwise2_x, 5)
        print("After Perm 2", x.shape)

        # x = self.oneXone(x)
        # print("oneXone", x.shape)

        x = self.pool(F.relu(x))
        # x = self.conv2(x)

        x = F.relu(self.onexone(x))

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        # print("After FC1", x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print("After FC3", x.shape)
        return x


    # def forward(self, x):
    #     # print("Before 1", x.shape)
    #     # 3 -> 18
    #     depthwise_x = self.depthwise(x)
    #     # print("After Depthwise 1", depthwise_x.shape)
    #     # x = F.relu(depthwise_x)
    #
    #     # 18 -> 16
    #     x = self.permuted_new_one_filter(depthwise_x, 6)
    #     # print("After Perm 1", x.shape)
    #
    #     x = self.pool(F.relu(x))
    #     # print("After Pool 1", x.shape)
    #
    #     # 16 -> 96
    #     depthwise2_x = self.depthwise2(x)
    #     # print("After Depthwise 2", depthwise2_x.shape)
    #
    #     # 96 -> 16
    #     x = self.permuted_new_one_filter(depthwise2_x, 6)
    #     print("After Perm 2", x.shape)
    #
    #     # x = self.oneXone(x)
    #     # print("oneXone", x.shape)
    #
    #     x = self.pool(F.relu(x))
    #     # x = self.conv2(x)
    #
    #     # x = F.relu(self.onexone(x))
    #
    #     x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    #     x = F.relu(self.fc1(x))
    #     # print("After FC1", x.shape)
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     # print("After FC3", x.shape)
    #     return x

    # def forward(self, x):
    #
    #     # print("Before 1", x.shape)
    #     # 3 -> 18
    #     depthwise_x = self.depthwise(x)
    #     # print("After Depthwise 1", depthwise_x.shape)
    #
    #     # 18 -> 6
    #     x = self.permuted_new(depthwise_x, 6)
    #     # print("After Perm 1", x.shape)
    #
    #     x = self.pool(F.relu(x))
    #     # print("After Pool 1", x.shape)
    #
    #     # 6 -> 36
    #     depthwise2_x = self.depthwise2(x)
    #     # print("After Depthwise 2", depthwise2_x.shape)
    #
    #     # 36 -> 6
    #     x = self.permuted_new(depthwise2_x, 6)
    #     # print("After Perm 2", x.shape)
    #
    #     x = self.pool(F.relu(x))
    #
    #     x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    #     x = F.relu(self.fc1(x))
    #     # print("After FC1", x.shape)
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     # print("After FC3", x.shape)
    #     return x

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

    if(epoch%5 == 0):
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