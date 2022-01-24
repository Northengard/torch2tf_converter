from torch import nn


class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        n_chn = 32
        self.conv1 = nn.Conv2d(3, n_chn, (3, 3), padding=1, stride=(1, 1), bias=False)
        self.conv1_bn = nn.BatchNorm2d(n_chn)
        self.conv1_relu = nn.ReLU()

        self.adp_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(32, 16, bias=False)
        self.d2 = nn.Linear(16, 10)
        self.d2_relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.adp_avg_pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d2_relu(x)
        return x
