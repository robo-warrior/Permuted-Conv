'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class permuted_SmallCNN(nn.Module):
    def __init__(self, num_out_channels=100):
        super(permuted_SmallCNN, self).__init__()
        self.num_layers = 2
        self.num_filters1 = 5
        self.num_filters2 = 10

        self.in_channels1 = 3
        self.num_constrained_filters1 = 5
        self.in_channels2 = 5
        self.num_constrained_filters2 = 10
        self.index_list = []

        temp = torch.randint(0, self.num_constrained_filters1, (self.num_constrained_filters1 * self.in_channels1,))
        for i in range(temp.shape[0]):
            temp[i] = temp[i] * self.in_channels1 + ((i+1) % self.in_channels1)
        self.index_list.append(temp)
        temp = torch.randint(0, self.num_constrained_filters2, (self.num_constrained_filters2 * self.in_channels2,))
        for i in range(temp.shape[0]):
            temp[i] = temp[i] * self.in_channels2 + ((i+1) % self.in_channels2)
        self.index_list.append(temp)


        # self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.dwconv1 = nn.Conv2d(in_channels=3, out_channels=self.num_filters1 * 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.onexone1 = nn.Conv2d(in_channels = self.num_filters1 + self.num_constrained_filters1, out_channels=self.num_filters1, kernel_size=1)
        # self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, bias=False, groups=self.num_filters1)
        self.dwconv2 = nn.Conv2d(in_channels=self.num_filters1, out_channels=self.num_filters2 * self.num_filters1, kernel_size=3, padding=1, groups=self.num_filters1, bias=False)
        self.onexone2 = nn.Conv2d(in_channels=self.num_filters2 + self.num_constrained_filters2, out_channels=self.num_filters2, kernel_size=1)
        self.fc = nn.Linear(8 * 8 * self.num_filters2, num_out_channels)

    def forward(self, x):
        out = self.dwconv1(x)

        # Original DW conv
        pos = [i for i in range(1, self.num_filters1 + 1)]
        ind = np.argsort(np.array(pos * (int)(out.shape[1]/self.num_filters1)))
        out = out[:, ind, :, :]

        # Adding permuted channels
        constrained_filter_channels = out[:, self.index_list[0], :, :]
        out = torch.cat((out, constrained_filter_channels), dim=1)

        # Groupwise summation across depth
        out = out.reshape(out.shape[0], self.num_filters1 + self.num_constrained_filters1, self.in_channels1, out.shape[2], out.shape[3]).sum(2)

        out = F.relu(self.onexone1(out))
        out = F.max_pool2d(out, 2)
        out = self.dwconv2(out)

        pos = [i for i in range(1, self.num_filters2 + 1)]
        ind = np.argsort(np.array(pos * (int)(out.shape[1] / self.num_filters2)))
        out = out[:, ind, :, :]


        # Adding permuted channels
        constrained_filter_channels = out[:, self.index_list[1], :, :]
        out = torch.cat((out, constrained_filter_channels), dim=1)

        # Groupwise summation across depth
        out = out.reshape(out.shape[0], self.num_filters2 + self.num_constrained_filters2, self.in_channels2,
                          out.shape[2], out.shape[3]).sum(2)

        out = F.relu(self.onexone2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
