from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.freeConvStack = nn.Sequential( #define the order in which data passes through net
            LocallyConnected2d(3, out_channels=32, output_size=(54,44), kernel_size=6, stride=4, bias=True),
            # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
            # weights = (kernel_size*kernel_size*out_channels+1(for bias))*output_size*output_size
            nn.ReLU(),
            
            LocallyConnected2d(32, 64, (13,10), 6, 4,True), # here input size = output size previous layer not out_channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),
            #nn.Linear() #how to calculate input size for this layer?
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        output = self.freeConvStack(x)
        output = output.softmax(dim=1)
        return output

