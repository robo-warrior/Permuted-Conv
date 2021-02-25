'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_out_channels=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet_1x1(nn.Module):
    def __init__(self, num_out_channels=100):
        super(LeNet_1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6*3, kernel_size=5, groups=3)
        self.onexone1 = nn.Conv2d(in_channels=6*3, out_channels=6, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6*16, kernel_size=5, groups=6)
        self.onexone2 = nn.Conv2d(in_channels=6*16, out_channels=16, kernel_size=1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_out_channels)

    def forward(self, x):
        out = F.relu(self.onexone1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.onexone2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

