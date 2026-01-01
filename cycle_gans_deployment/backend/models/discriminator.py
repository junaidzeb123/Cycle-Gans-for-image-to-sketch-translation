from __future__ import annotations

from torch import nn

class Discriminator(nn.Module):
      def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 3,   out_channels = 64, stride = 2, kernel_size = 4,  padding = 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = 64,  out_channels = 128, stride = 2, kernel_size = 4,  padding = 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),


            nn.Conv2d(in_channels = 128, out_channels = 256, stride = 2, kernel_size = 4,  padding = 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = 256, out_channels = 512, stride = 1, kernel_size=4, padding = 1 ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels = 512, out_channels = 1,   stride = 1, kernel_size=4,  padding = 1),
            nn.Flatten()
        )

      def forward(self, x):
          return self.network(x)
