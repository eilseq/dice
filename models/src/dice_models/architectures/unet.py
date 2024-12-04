import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention_g_channels, attention_l_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = AttentionBlock(
            F_g=attention_g_channels, F_l=attention_l_channels, F_int=out_channels)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        skip_connection = self.attention(g=x, x=skip_connection)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, num_channels):
        super(AttentionUNet, self).__init__()
        self.encoder1 = EncoderBlock(num_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)

        self.bridge = ConvBlock(128, 256)

        self.decoder1 = DecoderBlock(256 + 128, 128, 256, 128)
        self.decoder2 = DecoderBlock(128 + 64, 64, 128, 64)

        self.final_conv = nn.Conv2d(64, num_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)

        b1 = self.bridge(p2)

        d1 = self.decoder1(b1, s2)
        d2 = self.decoder2(d1, s1)

        outputs = self.final_conv(d2)
        return outputs
