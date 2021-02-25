'''Train CIFAR10 with PyTorch.'''
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar
import torch.optim.lr_scheduler as lrs

experiment = Experiment(api_key="hPc2DeBWvLYMqWUFLMgVTSQrF",
                        project_name="permuted-convolutions", workspace="rishabh")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# Ideas
# Pretrain network without permuted convolutions. Then train it using permuted/shuffled convolutions
################################################
num_channels_permuted = "0"
model_name = "LeNet_1x1_regularized_conv1-2"
gpu_id = 3
reg_lambda = 5e-1
################################################

experiment.add_tag(model_name)
experiment.add_tag(num_channels_permuted)
experiment.log_other("Network", "LeNet_1x1_regularized")
experiment.log_other("Dataset", "CIFAR-100")
experiment.log_other("Type", "1x1_regularized_conv1-2")
experiment.log_other("Regularizer", reg_lambda)

device = 'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_batch_size = 100
test_batch_size = 100

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
################################################
# net = VGG('VGG19')

# net = ResNet18()
# net = PermResNet18()
# net = PermResNet_no_constraints18()
# net = ShuffledResNet18()
# net = ResNet18_1x1()
# net = ShuffledResNetNormalized18()
# net = PermResNet18_1x1_Dropout()
net = LeNet_1x1()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
################################################
net = net.to(device)

# if device == 'cuda:2':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/resnet18_reduced_ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
model_scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

print(net)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # ResNet-18_1x1:
        # loss_reg = reg_lambda * (
        #         # torch.norm(net.layer1[0].onexone1.weight.data, 1) + torch.norm(net.layer1[0].onexone2.weight.data, 1)
        #         # + torch.norm(net.layer1[1].onexone1.weight.data, 1) + torch.norm(net.layer1[1].onexone2.weight.data, 1)
        #         torch.norm(net.layer2[0].onexone1.weight.data, 1) + torch.norm(net.layer2[0].onexone2.weight.data, 1)
        #         + torch.norm(net.layer2[1].onexone1.weight.data, 1) + torch.norm(net.layer2[1].onexone2.weight.data, 1)
        #         + torch.norm(net.layer3[0].onexone1.weight.data, 1) + torch.norm(net.layer3[0].onexone2.weight.data, 1)
        #         + torch.norm(net.layer3[1].onexone1.weight.data, 1) + torch.norm(net.layer3[1].onexone2.weight.data, 1)
        #         # + torch.norm(net.layer4[0].onexone1.weight.data, 1) + torch.norm(net.layer4[0].onexone2.weight.data, 1)
        #         # + torch.norm(net.layer4[1].onexone1.weight.data, 1) + torch.norm(net.layer4[1].onexone2.weight.data, 1)
        # )

        # LeNet-5_1x1
        loss_reg = reg_lambda * (
                torch.norm(net.onexone1.weight.data, 1) + torch.norm(net.onexone2.weight.data, 1)
        )
        loss = criterion(outputs, targets) + loss_reg
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    model_scheduler.step(train_loss/len(trainloader))
    return train_loss/len(trainloader), 100.*correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # ResNet-18_1x1
            # loss_reg = reg_lambda * (
            #     # torch.norm(net.layer1[0].onexone1.weight.data, 1) + torch.norm(net.layer1[0].onexone2.weight.data, 1)
            #     # + torch.norm(net.layer1[1].onexone1.weight.data, 1) + torch.norm(net.layer1[1].onexone2.weight.data, 1)
            #         torch.norm(net.layer2[0].onexone1.weight.data, 1) + torch.norm(net.layer2[0].onexone2.weight.data,                                                                   1)
            #         + torch.norm(net.layer2[1].onexone1.weight.data, 1) + torch.norm(net.layer2[1].onexone2.weight.data,                                                                     1)
            #         + torch.norm(net.layer3[0].onexone1.weight.data, 1) + torch.norm(net.layer3[0].onexone2.weight.data,                                                                  1)
            #         + torch.norm(net.layer3[1].onexone1.weight.data, 1) + torch.norm(net.layer3[1].onexone2.weight.data,                                                                    1)
            #         # + torch.norm(net.layer4[0].onexone1.weight.data, 1) + torch.norm(net.layer4[0].onexone2.weight.data, 1)
            #         # + torch.norm(net.layer4[1].onexone1.weight.data, 1) + torch.norm(net.layer4[1].onexone2.weight.data, 1)
            # )
            # LeNet-5_1x1
            loss_reg = reg_lambda * (
                torch.norm(net.onexone1.weight.data, 1) + torch.norm(net.onexone2.weight.data, 1)
            )
            loss = criterion(outputs, targets) + loss_reg
            # loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/shuffledresnet18_layernorm_no_params_ckpt.pth')
        best_acc = acc

    return test_loss / len(testloader), 100.*correct/total

training_acc_list = []
testing_acc_list = []
training_loss_list = []
testing_loss_list = []

best_train_acc = 0
best_test_acc = 0
for epoch in range(start_epoch, start_epoch+3000):
    print(model_name)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    training_loss_list.append(train_loss)
    testing_loss_list.append(test_loss)
    training_acc_list.append(train_acc)
    testing_acc_list.append(test_acc)
    if(train_acc > best_train_acc):
        best_train_acc = train_acc
    if (test_acc > best_test_acc):
        best_test_acc = test_acc

    experiment.log_metric("best_train_acc", best_train_acc, epoch=epoch+1)
    experiment.log_metric("best_test_acc", best_test_acc, epoch=epoch + 1)
    experiment.log_metric("train_acc", train_acc, epoch=epoch + 1)
    experiment.log_metric("test_acc", test_acc, epoch=epoch + 1)

    plt.plot(training_loss_list, color='blue', label='Training')
    plt.plot(testing_loss_list, color='red', label='Testing', alpha=.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.legend()
    # plt.savefig("./loss_plot_perm2.png", format='png')
    experiment.log_figure(figure=plt, figure_name='loss_plot', overwrite=True)
    plt.close()

    plt.plot(training_acc_list, color='blue', label='Training')
    plt.plot(testing_acc_list, color='red', label='Testing', alpha=.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot')
    plt.legend()
    # plt.savefig("./accuracy_plot_perm2.png", format='png')
    experiment.log_figure(figure=plt, figure_name='accuracy_plot', overwrite=True)
    plt.close()
