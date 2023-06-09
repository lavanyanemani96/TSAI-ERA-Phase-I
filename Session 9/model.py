# model.py file 

import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3), padding = 0, stride = 1, dilation = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3), padding = 0, stride = 1, dilation = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            depthwise_separable_conv(64, 64),
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding = 2, stride = 1, dilation = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding = 2, stride = 1, dilation = 2, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3), padding = 0, stride = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3), padding = 0, stride = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=16),
            nn.Conv2d(in_channels=64,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)+x
        x = self.convblock3(x)+x
        x = self.convblock4(x)

        x = self.gap(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)