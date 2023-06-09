from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.freeConvStack = nn.Sequential( #define the order in which data passes through net
            LocallyConnected2d(3, out_channels=1, output_size=(107,87), kernel_size=6, stride=2, bias=True),
            # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
            # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
            nn.ReLU(),
            
            LocallyConnected2d(1, 1, (102,82), 6, 1,True), # here input size = output size previous layer not out_channels
            nn.ReLU(),

            LocallyConnected2d(1, 1, (97,77), 6, 1, True),
            nn.ReLU(),

            LocallyConnected2d(1, 1, (92,72), 6, 1, True),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(1*46*36, 300), # input size = output_channels*(output_size/2) of last Local layer because of maxpool2d kernel size 2
            nn.ReLU(),
            nn.Linear(300, 200)
        )

    def forward(self, x):
        output = self.freeConvStack(x)
        return output