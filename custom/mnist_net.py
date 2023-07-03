from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        self.Seq_stack = nn.Sequential(
        LocallyConnected2d(1, out_channels=16, output_size=13, kernel_size=3, stride=2, bias=True),
        # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
        # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
        nn.ReLU(),
        LocallyConnected2d(16, 32, 6, 3, 2,True), # here input size = output size previous layer not out_channels
        nn.ReLU(),
        LocallyConnected2d(32, 64, 4, 3, 1,True), # here input size = output size previous layer not out_channels
        
        nn.Flatten(),

        nn.Linear(64*4*4, 512), # input size = output_channels*output_size/2 of last Local layer because of maxpool2d kernel size 2
        nn.ReLU(),
        nn.Linear(512, 10),
        )
        



    def forward(self, x):
        output = self.Seq_stack(x)
        return output