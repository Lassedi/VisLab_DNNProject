from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.LL1 = LocallyConnected2d(1, out_channels=16, output_size=13, kernel_size=3, stride=2, bias=True)
        # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
        # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
        
        self.LL2 = LocallyConnected2d(16, 32, 6, 3, 2,True) # here input size = output size previous layer not out_channels
        self.LL3 = LocallyConnected2d(32, 64, 4, 3, 1,True) # here input size = output size previous layer not out_channels
        
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(64*4*4, 512) # input size = output_channels*output_size/2 of last Local layer because of maxpool2d kernel size 2
        self.linear2 = nn.Linear(512, 10)
        self.activation = nn.ReLU()
        self.softm = nn.Softmax(dim=-1)



    def forward(self, x):
        output = self.LL1(x)
        output = self.activation(output)
        output = self.LL2(output)
        output = self.activation(output)
        output = self.LL3(output)
        output = self.activation(output)

        output = self.flatten(output)
        output = self.linear1(output)
        output = self.activation(output)
        output = self.linear2(output)
        output = self.softm(output)
        return output