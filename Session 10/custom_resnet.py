# model.py file 

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()

        self.preparation = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

        self.layer1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(128),
                nn.ReLU()
                )

        self.residual1 = block(128, 128, 1)

        self.layer2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )

        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU()
                )

        self.residual3 = block(512, 512, 1)

        self.maxpool2d = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.preparation(x)

        x = self.layer1(x)
        res1 = self.residual1(x)
        x = x + res1

        x = self.layer2(x)

        x = self.layer3(x)
        res3 = self.residual3(x)
        x = x + res3

        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def Custom_ResNet():
    return ResNet(BasicBlock, num_classes=10)