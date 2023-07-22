from torch import nn 

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_stack = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=3, bias=True),
            nn.Conv2d(32,64, 4, 3, bias= True),
            nn.Conv2d(64,64, 3, 2, bias=True),
            nn.Flatten(),
            nn.Linear(64*11*9, 10177)

        )

    def forward(self, x):
        output = self.sequential_stack(x)
        return output