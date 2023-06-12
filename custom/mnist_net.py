from torch import nn
from custom.LocallyConnected2d import LocallyConnected2d

class FreeConvNetwork(nn.Module):
    def __init__(self):
        super().__init__() # call upon parent class constructor ie their attributes and methods 
        #self.freeConvStack = nn.Sequential( #define the order in which data passes through net
        self.LL1 = LocallyConnected2d(1, out_channels=32, output_size=(13,13), kernel_size=3, stride=2, bias=True)
        # output_size = ((input size(width/height of img) - kernel_size + 2xpadding) / stride) + 1(for bias)
        # weights = (kernel_size*kernel_size*in_channels+1(for bias))*out_channels*output_size*output_size
        self.activation = nn.ReLU()
        
        self.LL2 = LocallyConnected2d(32, 64, (6,6), 3, 2,True) # here input size = output size previous layer not out_channels
        self.LL3 = LocallyConnected2d(64, 128, (4,4), 3, 1,True) # here input size = output size previous layer not out_channels

        # self.LL4 = LocallyConnected2d(3, 3, (46,36), 3, 1,True) # here input size = output size previous layer not out_channels
        # self.LL5 = LocallyConnected2d(3, 3, (44,34), 3, 1,True) # here input size = output size previous layer not out_channels
        # self.LL6 = LocallyConnected2d(3, 3, (42,32), 3, 1,True) # here input size = output size previous layer not out_channels
        
        # self.LL7 = LocallyConnected2d(3, 3, (20,15), 2, 1,True)
        # self.LL8 = LocallyConnected2d(3, 3, (19,14), 2, 1,True)
        # self.LL9 = LocallyConnected2d(3, 3, (18,13), 2, 1,True)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.batchNorm = nn.BatchNorm2d()
        
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128*4*4, 10) # input size = output_channels*output_size/2 of last Local layer because of maxpool2d kernel size 2


        # self.linear2 = nn.Linear(1000, 200)
        # self.linear3 = nn.Linear(500,200)
        #)


    def forward(self, x):
        a = self.LL1(x)
        a = self.activation(a)
        b = self.LL2(a)
        b = self.activation(b)
        c = self.LL3(b)
        c = self.activation(c)

        # xx = self.maxpool(c)

        # d = self.LL4(xx)
        # d = self.activation(d)
        # f = self.LL5(d)
        # f = self.activation(f)
        # g = self.LL6(f)
        # g = self.activation(g)
        
        # e = self.maxpool(g)

        # e = self.LL7(e)
        # e = self.LL8(e)
        # e = self.LL9(e)

        # e = self.maxpool(e)

        e = self.flatten(c)
        e = self.linear1(e)
        e = self.activation(e)
        # e = self.linear2(e)
        # e = self.activation(e)
        # e = self.linear3(e)
        # e = self.activation(e)

        return e