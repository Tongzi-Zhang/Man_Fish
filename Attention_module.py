import torch
from torch import nn
from model.Res_block import ResidualBlock, MCNN_layer, Attention_two_conv, ResidualBlock1, ResidualBlock2, ResidualBlock3


class Attention_stage1(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage1, self).__init__()

        self.out_mpool = nn.MaxPool2d(kernel_size=2)
        self.out_softmax = ResidualBlock(in_channels, out_channels)
        self.out_skip_connection = ResidualBlock(in_channels, out_channels)
        self.out_interp = nn.UpsamplingBilinear2d(scale_factor=2)
        # trunk
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.out_trunk1 = MCNN_layer(out_channels, out_channels, ksize, padding, stride)

        self.out_softmax1 = Attention_two_conv(out_channels, ksize, stride, padding)


    def forward(self, input):
        out_mpool1 = self.out_mpool(input)
        out_softmax1 = self.out_softmax(out_mpool1)
        out_skip_connection = self.out_softmax(out_softmax1)
        out_mpool2 = self.out_softmax(out_softmax1)
        out_softmax2 = self.out_mpool(out_mpool2)
        out_softmax2 = self.out_softmax(out_softmax2)
        out_interp2 = self.out_interp(out_softmax2) + out_softmax1
        out1 = out_interp2 + out_skip_connection
        out_softmax3 = self.out_softmax(out1)
        out_interp1 = self.out_interp(out_softmax3)
        out_trunk = self.out_trunk(input)
        out_trunk = self.out_trunk1(out_trunk)
        out2 = out_interp1 + out_trunk

        out_softmax4 = self.out_softmax1(out2)
        last_out = (1 + out_softmax4) * out_trunk

        return last_out

class Attention_stage2(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage2, self).__init__()
        self.out_mpool = nn.MaxPool2d(kernel_size=2)
        self.out_softmax = ResidualBlock(in_channels, out_channels)
        self.out_interp = nn.UpsamplingBilinear2d(scale_factor=2)
        # trunk
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.out_trunk1 = MCNN_layer(out_channels, out_channels, ksize, padding, stride)

        self.out_softmax1 = Attention_two_conv(out_channels, ksize, stride, padding)

    def forward(self, input):
        out_mpool1 = self.out_mpool(input)
        out_softmax1 = self.out_softmax(out_mpool1)
        out_softmax1 = self.out_softmax(out_softmax1)
        out_interp = self.out_interp(out_softmax1)

        out_trunk = self.out_trunk(input)
        out_trunk = self.out_trunk1(out_trunk)

        out1 = out_interp + out_trunk

        out_softmax2 = self.out_softmax1(out1)
        last_out = (1 + out_softmax2) * out_trunk

        return last_out

class Attention_stage3(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage3,self).__init__()
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.res = ResidualBlock1(in_channels, out_channels)

    def forward(self, input):
        out_trunk = self.out_trunk(input)
        res = self.res(input)
        out_res = torch.sigmoid(res)
        out_atten = (1 + out_res) * out_trunk
        return out_atten

class Attention_stage4(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage4,self).__init__()
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.res = ResidualBlock(in_channels, out_channels)

    def forward(self, input):
        out_trunk = self.out_trunk(input)
        res = self.res(input)
        out_res = torch.sigmoid(res)
        out_atten = (1 + out_res) * out_trunk
        return out_atten

class Attention_stage5(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage5,self).__init__()
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.res = ResidualBlock2(in_channels, out_channels)

    def forward(self, input):
        out_trunk = self.out_trunk(input)
        res = self.res(input)
        out_res = torch.sigmoid(res)
        out_atten = (1 + out_res) * out_trunk
        return out_atten

class Attention_stage6(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(Attention_stage6,self).__init__()
        self.out_trunk = MCNN_layer(in_channels, out_channels, ksize, padding, stride)
        self.res = ResidualBlock3(in_channels, out_channels)

    def forward(self, input):
        out_trunk = self.out_trunk(input)
        res = self.res(input)
        out_res = torch.sigmoid(res)
        out_atten = (1 + out_res) * out_trunk
        return out_atten
