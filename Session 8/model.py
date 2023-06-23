# model.py file 

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization_layer(n, norm_arg):

    if n == "BN":
        return nn.BatchNorm2d(norm_arg[0])
    elif n == "LN":
        return nn.GroupNorm(1, norm_arg[0])
    elif n == "GN":
        return nn.GroupNorm(norm_arg[0]//2, norm_arg[0])
    else:
        raise ValueError('Valid options are BN/LN/GN')

dropout_value = 0.005

class Net(nn.Module):
    def __init__(self, n_type='BN'):
        self.n_type = n_type
        super(Net,self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 30, 30])
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 28, 28])
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 28, 28])
        )

        # Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 12, 12])
        )

        # Convolution Block 5
        self.convblock5  = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 10, 10])
        )

        # Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 10, 10])
        )

        # Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )


        # Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 5, 5])
        )

        # Convolution Block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 5, 5])
        )

        # Convolution Block 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [32, 5, 5])
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )

        # Convolution Block 10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=10,kernel_size=(1,1),padding = 0, bias = False)
        )

    def forward(self, x):
        # C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        x = self.transblock1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x + self.convblock6(x)

        x = self.transblock2(x)

        x = x + self.convblock7(x)
        x = x + self.convblock8(x)
        x = x + self.convblock9(x)

        x = self.gap(x)
        x = self.convblock10(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)