from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.freeConvStack = nn.Sequential( #define the order in which data passes through net
            nn.BatchNorm2d(3),
            LocallyConnected2d(in_channels=3, out_channels=32, output_size=(54,54), 
                               kernel_size=10, stride=4, bias=True),
            # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
            # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2, padding=1),

            LocallyConnected2d(32,96, (23,23),5, 1,True), # here input size = output size previous layer not out_channels
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3,2, padding=1),

            LocallyConnected2d(96, 192, (10,10), 3, 1, True),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            LocallyConnected2d(192, 128, (8,8), 3, 1, True),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            LocallyConnected2d(128, 128, (6,6), 3, 1, True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1),
        )
        self.classifier_stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            
            nn.Linear(128*2*2, 4096), # input size = output_channels*(output_size/2) of last Local layer because of maxpool2d kernel size 2
            #nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            #nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Linear(4096, 500),
        )

    def forward(self, x):
        output = self.freeConvStack(x)
        output = self.classifier_stack(output)
        return output

