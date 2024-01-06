import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.conv1 = nn.Sequential(  #input_size = 32
            nn.Conv2d(64, 128, 3, padding=1, stride = 1, bias=False), #kernel_is: 3x3
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.r1 = nn.Sequential( #16
            nn.Conv2d(128, 128, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) #16
        # Conv 3x3 [256k] LAYER2
        self.conv2 = nn.Sequential( #16
            nn.Conv2d(128, 256, 3, padding = 1, stride = 1, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ) #8
        # LAYER3 X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride = 1, bias=False), #kernel_is: 3x3
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) #4
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.r2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(in_features=512, out_features=10) #cifar10, so 10
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.prep(x)
        x = self.conv1(x)
        r1 = self.r1(x)
        x = x + r1
        x = self.conv2(x)
        x = self.conv3(x)
        r2 = self.r2(x)
        x = x + r2
        x = self.gap(x)
        # print("Ater pool shape, ", x.shape) #torch.Size([2, 512, 1, 1])
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)