'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SmallCNN(nn.Module):
    def __init__(self, num_out_channels=100):
        super(SmallCNN, self).__init__()
        self.num_filters1 = 5
        self.num_filters2 =10

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2, kernel_size=3, padding=1)
        self.fc = nn.Linear(8*8*self.num_filters2, num_out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SmallCNN_weighted(nn.Module):
    def __init__(self, num_out_channels=100):
        super(SmallCNN_weighted, self).__init__()
        self.num_filters1 = 5
        self.num_filters2 = 10

        # self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.onexone1 = nn.Conv2d(in_channels=self.num_filters1 * 3, out_channels=self.num_filters1, kernel_size=1, groups=self.num_filters1)
        # self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, bias=False, groups=self.num_filters1)
        self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, groups=self.num_filters1, bias=False)
        self.onexone2 = nn.Conv2d(in_channels=self.num_filters2 * self.num_filters1, out_channels=self.num_filters2, kernel_size=1, groups=self.num_filters2)
        self.fc = nn.Linear(8 * 8 * self.num_filters2, num_out_channels)

    def forward(self, x):
        out = self.dwconv1(x)

        pos = [i for i in range(1, self.num_filters1 + 1)]
        ind = np.argsort(np.array(pos * (int)(out.shape[1]/self.num_filters1)))
        out = out[:, ind, :, :]

        out = F.relu(self.onexone1(out))
        out = F.max_pool2d(out, 2)
        out = self.dwconv2(out)

        pos = [i for i in range(1, self.num_filters2 + 1)]
        ind = np.argsort(np.array(pos * (int)(out.shape[1] / self.num_filters2)))
        out = out[:, ind, :, :]

        out = F.relu(self.onexone2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SmallCNN_1x1(nn.Module):
    def __init__(self, num_out_channels=100):
        super(SmallCNN_1x1, self).__init__()
        self.num_filters1 = 5
        self.num_filters2 = 10

        # self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, groups=3)
        self.onexone1 = nn.Conv2d(in_channels=self.num_filters1 * 3, out_channels=self.num_filters1, kernel_size=1, groups=self.num_filters1)
        # self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, bias=False, groups=self.num_filters1)
        self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, groups=self.num_filters1)
        self.onexone2 = nn.Conv2d(in_channels=self.num_filters2 * self.num_filters1, out_channels=self.num_filters2, kernel_size=1, groups=self.num_filters2)
        self.fc = nn.Linear(8 * 8 * self.num_filters2, num_out_channels)

    def forward(self, x):
        out = F.relu(self.onexone1(self.dwconv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.onexone2(self.dwconv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class E2ESmallCNN(nn.Module):
    def __init__(self, num_out_channels=100):
        super(E2ESmallCNN, self).__init__()
        self.num_filters1 = 5
        self.num_filters2 =10

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2, kernel_size=3, padding=1) #16 x 16 x 10
        self.conv3 = nn.Conv2d(in_channels=self.num_filters2, out_channels=1, kernel_size=7, padding=0) #10 x 10 x 1

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        return out


class E2ESmallCNN_1x1(nn.Module):
    def __init__(self, num_out_channels=100):
        super(E2ESmallCNN_1x1, self).__init__()
        self.num_filters1 = 5
        self.num_filters2 = 10
        self.num_filters3 = 1

        self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.onexone1 = nn.Conv2d(in_channels=self.num_filters1 * 3, out_channels=self.num_filters1, kernel_size=1, groups=self.num_filters1)
        self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, bias=False, groups=self.num_filters1)
        self.onexone2 = nn.Conv2d(in_channels=self.num_filters2 * self.num_filters1, out_channels=self.num_filters2, kernel_size=1, groups=self.num_filters2) # 16 x 16 x 10
        self.dwconv3 = nn.Conv2d(in_channels=self.num_filters2, out_channels=self.num_filters3 * self.num_filters2, kernel_size=7, padding=0, bias=False, groups=self.num_filters2) # 10 x 10 x 10
        self.onexone3 = nn.Conv2d(in_channels=self.num_filters3 * self.num_filters2, out_channels=self.num_filters3, kernel_size=1, groups=self.num_filters3) # 10 x 10 x 1


    def forward(self, x):
        out = F.relu(self.onexone1(self.dwconv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.onexone2(self.dwconv2(out)))
        out = F.relu(self.onexone3(self.dwconv3(out)))
        out = out.view(out.size(0), -1)
        return out
