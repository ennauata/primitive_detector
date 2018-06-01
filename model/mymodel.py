import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, drn_base):
        super(MyModel, self).__init__()
        self.drn_base = drn_base
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.reg = nn.Conv2d(128, 3, 1, stride=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.seg = nn.Conv2d(128, 1, 1, stride=1)

    def forward(self, x):

        # extract features
        x = self.drn_base(x)

        # upsample
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)

        # compute detections
        y = self.conv1(x)
        y = self.relu3(y)
        det = self.reg(y)

        # upsample
        z = self.deconv3(x)
        z = self.relu4(z)

        # compute segments
        z = self.conv2(z)
        z = self.relu5(z)
        seg = self.seg(z)

        return det, seg