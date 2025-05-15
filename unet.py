import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        vgg = models.vgg16_bn(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:23])  # Until conv4_3

    def forward(self, x):
        return self.encoder(x)

class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.up(x)
    

class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
        


    