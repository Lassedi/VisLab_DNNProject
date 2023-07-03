from torch import nn 

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, bias=True),
            nn.Conv2d(32,64, 3, 2, bias= True),
            nn.Conv2d(64,128, 3, 1, bias=True),
            nn.Flatten(),
            nn.Linear(128*4*4, 200)
        )

    def forward(self, x):
        output = self.sequential_stack(x)
        return output