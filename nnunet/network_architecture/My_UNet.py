import torch
import torch.nn as nn

from nnunet.network_architecture.neural_network import SegmentationNetwork


# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_ch, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_ch, momentum=0.1, affine=True, track_running_stats=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv_block(in_ch, out_ch)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False, padding=0)
        self.conv = conv_block(out_ch + out_ch, out_ch)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class build_unet(SegmentationNetwork):
    def __init__(self, deep_supervision=False):
        super().__init__()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        """ Encoder """
        self.e1 = encoder_block(4, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)

        """ Bottleneck """
        self.b = conv_block(256, 320)

        """ Decoder """
        self.d1 = decoder_block(320, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv3d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs
