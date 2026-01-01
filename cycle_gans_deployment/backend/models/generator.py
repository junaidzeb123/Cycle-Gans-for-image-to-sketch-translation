from __future__ import annotations

from torch import nn

class ResidualBlock(nn.Module):
      def __init__(self):
        super(ResidualBlock, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, padding = 1 , kernel_size = 3),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, padding = 1 , kernel_size = 3),
            nn.InstanceNorm2d(256)
        )

      def forward(self, x):
          return x +  self.network(x)

class Generator(nn.Module):

      def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
           nn.Conv2d(in_channels = 3,   out_channels = 64,  kernel_size = 7, padding = 3, stride = 1),
           nn.InstanceNorm2d(64),
           nn.ReLU(True),

           nn.Conv2d(in_channels = 64,  out_channels = 128, kernel_size = 3, padding = 1, stride = 2),
           nn.InstanceNorm2d(128),
           nn.ReLU(True),

           nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 2),
           nn.InstanceNorm2d(256),
           nn.ReLU(True),
        )

        self.transformer = nn.Sequential(
          ResidualBlock(),
          ResidualBlock(),
          ResidualBlock(),
          ResidualBlock(),
          ResidualBlock(),
          ResidualBlock()
        )

        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(in_channels = 256, out_channels = 128, stride = 2, kernel_size = 3, padding = 1, output_padding = 1 ),
          nn.ConvTranspose2d(in_channels = 128, out_channels = 64,  stride = 2, kernel_size = 3, padding = 1, output_padding = 1 ),
          nn.ConvTranspose2d(in_channels = 64,  out_channels = 3,   stride = 1, kernel_size = 7, padding = 3),
          nn.Tanh()

        )

      def forward(self, x):
          x = self.encoder(x)
          x = self.transformer(x)
          x = self.decoder(x)
          return x
