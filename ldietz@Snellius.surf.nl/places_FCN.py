from custom.LocallyConnected2d import LocallyConnected2d
from torch import nn

class FreeConvNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Sequential_stack = nn.Sequential(
            LocallyConnected2d(3, 1, (251,251), 6, 1,True),
            nn.ReLU(),
            nn.MaxPool2d(6,2),
            
            LocallyConnected2d(1, 4, (59,59), 6, 2, True),
            nn.ReLU(),
            LocallyConnected2d(4, 8, (18,18), 6, 3, True),
            nn.ReLU(),
            # LocallyConnected2d(8, 8, (18,18), 6, 3, True),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 365)
        )

    def forward(self, x):
        output = self.Sequential_stack(x)
        return output
    