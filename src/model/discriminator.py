import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c64 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.c128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.c256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.c512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x64 = self.c64(x)
        x128 = self.c128(x64)
        x256 = self.c256(x128)
        x512 = self.c512(x256)
        return self.final(x512)
