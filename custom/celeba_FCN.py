from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.freeConvStack = nn.Sequential( #define the order in which data passes through net
            LocallyConnected2d(in_channels=3, out_channels=8, output_size=(213, 173), kernel_size=6, stride=1, bias=True),
            # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
            # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
            nn.ReLU(),
            nn.MaxPool2d(2),

            LocallyConnected2d(8,16, (51,41),6, 2,True), # here input size = output size previous layer not out_channels
            nn.ReLU(),

            LocallyConnected2d(16, 32, (16, 12), 6, 3, True),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(32*16*12, 10178), # input size = output_channels*(output_size/2) of last Local layer because of maxpool2d kernel size 2
        )

    def forward(self, x):
        output = self.freeConvStack(x)
        return output