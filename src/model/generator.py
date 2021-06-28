import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c64 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
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
        self.c512_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.c512_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.c512_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.c512_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.c512_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x64 = self.c64(x)
        #         print(x64.shape)
        x128 = self.c128(x64)
        x256 = self.c256(x128)
        x512_1 = self.c512_1(x256)
        x512_2 = self.c512_2(x512_1)
        x512_3 = self.c512_3(x512_2)
        x512_4 = self.c512_4(x512_3)
        x512_5 = self.c512_5(x512_4)
        return x512_5, x512_4, x512_3, x512_2, x512_1, x256, x128, x64


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1
        self.c512_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # 2x2
        self.c512_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # 4x4
        self.c512_3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # 8x8
        self.c512_4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 16x16
        self.c512_5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 32x32
        self.c256 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 64x64
        self.c128 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 128x128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: tuple):
        x512_1 = self.c512_1(x[0])
        #         import pdb; pdb.set_trace()
        #         print('dec', x512_1.shape, x[1].shape)
        x512_2 = self.c512_2(torch.cat((x512_1, x[1]), dim=1))

        x512_3 = self.c512_3(torch.cat((x512_2, x[2]), dim=1))
        x512_4 = self.c512_4(torch.cat((x512_3, x[3]), dim=1))
        x512_5 = self.c512_5(torch.cat((x512_4, x[4]), dim=1))
        #         import pdb; pdb.set_trace()
        x256 = self.c256(torch.cat((x512_5, x[5]), dim=1))
        x128 = self.c128(torch.cat((x256, x[6]), dim=1))
        return self.final(x128)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
