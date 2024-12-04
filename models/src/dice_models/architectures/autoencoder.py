import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, num_channels):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3,
                      stride=2, padding=1),  # Output: 8x8
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),  # Output: 4x4
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Output: 8x8
            nn.ReLU(),

            nn.ConvTranspose2d(64, num_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),  # Output: 16x16
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
