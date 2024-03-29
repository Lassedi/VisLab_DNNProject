from torch import nn 

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_stack = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 11, stride=4, bias=True, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Conv2d(64, 192, 5, 2, bias= True, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Conv2d(192, 384, 3, 1, bias=True, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, bias=True, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.classifier_stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 6348),
        )

    def forward(self, x):
        output = self.feature_stack(x)
        output = self.classifier_stack(output)
        return output