import torch
import torch.nn as nn


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=(3, 3, 3), padding=1, bias=False, batch_norm=False):
        super(Residual3DBlock, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               kernel_size=kernel_size, padding=padding, bias=bias)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=kernel_size, padding=padding, bias=bias)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet_3DConv(nn.Module):
    def __init__(self, in_channels, internal_depth, blocks, kernel_size, stride, padding, bias, batch_norm):
        super(ResNet_3DConv, self).__init__()
        self.in_channels = 32
        self.layer_first_non_residual = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=internal_depth,
                                                                kernel_size=kernel_size, padding=padding, stride=stride,
                                                                bias=bias),
                                                      nn.ReLU(inplace=True))
        self.sequential_blocks = []
        for i in range(blocks):
            self.sequential_blocks.append(Residual3DBlock(in_channels=internal_depth, out_channels=internal_depth,
                                                          kernel_size=kernel_size, padding=padding, stride=stride,
                                                          bias=bias))
        self.layer_main_body = nn.Sequential(*self.sequential_blocks)
        self.conv_for_image = nn.Conv3d(in_channels=32, out_channels=1, stride=stride,
                                        kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        out = self.layer_first_non_residual(x)
        for block in self.sequential_blocks:
            out = block(out)
        out = self.conv_for_image(out)
        return out
