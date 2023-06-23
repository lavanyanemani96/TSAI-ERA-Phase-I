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
        
class Net_C(nn.Module):
    def __init__(self):
        
        super(Net_C, self).__init__()
        self.dropout_value = 0.0
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(self.dropout_value)
        ) # output_size = 10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(self.dropout_value)
        ) # output_size = 8
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        

class Net_B(nn.Module):
    def __init__(self):
        
        super(Net_B, self).__init__()
        self.dropout_value = 0.0
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 8
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class Net_A(nn.Module):
    def __init__(self):
        
        super(Net_A, self).__init__()
        self.dropout_value = 0.05 
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 26, Rin=1, K=3, S=1, Jin=1, Jout=1, Rout=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 24, Rin=3, K=3, S=1, Jin=1, Jout=1, Rout=5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24, Rin=5, K=1, S=1, Jin=1, Jout=1, Rout=5
        self.pool1 = nn.MaxPool2d(2, 2) 
        # output_size = 12, Rin=5, K=2, S=2, Jin=1, Jout=2, Rout=6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_value)
        ) # output_size = 10, Rin=6, K=3, S=1, Jin=2, Jout=2, Rout=10
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 8, Rin=10, K=3, S=1, Jin=2, Jout=2, Rout=12
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6, Rin=12, K=3, S=1, Jin=2, Jout=2, Rout=14
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.dropout_value)
        ) # output_size = 6, Rin=14, K=3, S=1, Jin=2, Jout=2, Rout=16

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, Rin=16, K=6, S=1, Jin=2, Jout=2, Rout=26
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, Rin=26, K=1, S=1, Jin=2, Jout=2, Rout=26

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
        
class Net_6(nn.Module):
  
  def __init__(self):
    super(Net_6, self).__init__()
    
    self.block1 = nn.Sequential(
        nn.Conv2d(1, 16, 3),                                                    # (28, 28, 1)  --> (3, 3, 1, 16)  --> (26, 26, 16)  [R.F. 3]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
        nn.Dropout2d(0.1), 

        nn.Conv2d(16, 16, 3),                                                   # (26, 26, 16) --> (3, 3, 16, 16) --> (24, 24, 16)  [R.F. 5]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
        nn.Dropout2d(0.1),

        nn.Conv2d(16, 32, 3),                                                   # (24, 24, 16) --> (3, 3, 16, 32) --> (22, 22, 32)  [R.F. 7]
        nn.ReLU(), 
        nn.BatchNorm2d(32), 
        nn.Dropout2d(0.1)
        )
    
    self.block2 = nn.Sequential(
        nn.Conv2d(32, 16, 1),                                                   # (22, 22, 32) --> (1, 1, 32, 16) --> (22, 22, 16)  [R.F. 7]
        nn.ReLU(),
        nn.MaxPool2d(2,2),                                                      # (22, 22, 16) --> (2, 2)         --> (11, 11, 16)  [R.F. 14]

        nn.Conv2d(16, 16, 3),                                                   # (11, 11, 16) --> (3, 3, 16, 16) --> (9, 9, 16)    [R.F. 16]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
        nn.Dropout2d(0.1)
    ) 

    self.block3 = nn.Sequential(
        nn.Conv2d(16, 16, 3),                                                   # (9, 8, 16)  --> (3, 3, 16, 16)  --> (7, 7, 16)     [R.F. 18]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
        nn.Dropout2d(0.1),
        
       nn.Conv2d(16, 16, 3, padding=1),                                        # (7, 7, 16)  --> (3, 3, 16, 16)  --> (7, 7, 16)     [R.F. 20]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
        nn.Dropout2d(0.1)
    )

    self.block4 = nn.Sequential(
        nn.Conv2d(16, 16, 3),                                                   # (7, 7, 16)  --> (3, 3, 16, 16)  --> (5, 5, 16)     [R.F. 22]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 

        nn.Conv2d(16, 16, 3, padding=1),                                        # (5, 5, 16)  --> (3, 3, 16, 16)  --> (5, 5, 16)     [R.F. 24]
        nn.ReLU(), 
        nn.BatchNorm2d(16), 
    ) 
    
    self.globalaveragepooling = nn.Sequential(
        nn.Conv2d(16, 10, 1),                                                   # (5, 5, 16)  --> (1, 1, 16, 10)  --> (5, 5, 10)     [R.F. 26]
        nn.AvgPool2d(5)
    )

    self.fc = nn.Sequential(
        nn.Linear(in_features=10, out_features=10)                              # Fully-connected network
    ) 

  def forward(self, x):

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.globalaveragepooling(x)

    x = x.view(-1, 10)
    x = self.fc(x)

    return F.log_softmax(x)
