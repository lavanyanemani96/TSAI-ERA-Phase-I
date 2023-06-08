# model.py file 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    
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

     
