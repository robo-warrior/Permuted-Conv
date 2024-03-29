'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class PermBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PermBasicBlock, self).__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = nn.Conv2d(
            in_planes, in_planes * planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=self.in_planes)
        self.onexone1 = nn.Conv2d(in_planes * planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * planes, kernel_size=3, stride=1, padding=1, bias=False, groups=self.planes)
        self.onexone2 = nn.Conv2d(planes * planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, in_planes * self.expansion*planes,
                #           kernel_size=1, stride=stride, bias=False, groups=in_planes),
                # nn.Conv2d(in_planes * self.expansion*planes, self.expansion*planes, kernel_size=1, groups=self.expansion*planes),
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.onexone1(self.conv1(x))))
        out = self.bn2(self.onexone2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):  # num_classes=100
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(PermBasicBlock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(PermBasicBlock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(PermBasicBlock, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(PermBasicBlock, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_multiple_1x1():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34_multiple_1x1():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50_multiple_1x1():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101_multiple_1x1():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152_multiple_1x1():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18_multiple_1x1()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
