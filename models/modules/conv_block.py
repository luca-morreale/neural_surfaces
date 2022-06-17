
from torch.nn import Conv2d
from torch.nn import Identity
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Upsample


class ConvBlock(Module):
    ## Single convolution layer for CNN in Neural Convolutional Surfaces

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, act_fun, act_args):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv     = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.act      = act_fun(**act_args)


    def forward(self, x):
        out1 = self.upsample(x)
        out2 = self.conv(out1)
        out  = self.act(out2)
        return out


class ResidualConvBlock(Module):
    ## Single residual convolution layer for CNN in Neural Convolutional Surfaces

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, act_fun, act_args):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.residual = Sequential(
                Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                act_fun(**act_args),
                Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        )

        # if input channels is different than output than use convolution
        self.shortcut = Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.post_act = act_fun(**act_args)


    def forward(self, x):
        out1  = self.upsample(x)
        short = self.shortcut(out1)
        res   = self.residual(out1)
        out   = res + short
        return out

