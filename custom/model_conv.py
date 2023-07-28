from torch import nn 

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_stack = nn.Sequential(
            nn.Conv2d(3, 4, 6, stride=3, bias=True),
            nn.Conv2d(4,16, 4, 3, bias= True),
            nn.Conv2d(16,32, 3, 2, bias=True),
            nn.Flatten(),
            nn.Linear(32*11*9, 10178)
        )

    def forward(self, x):
        output = self.sequential_stack(x)
        return output