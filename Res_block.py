import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    '''
    implement the residual module
    '''
    def __init__(self, inchannels, outchannels, stride=1):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outchannels, outchannels, kernel_size=1, padding=0, stride=stride)
        self.bn3 = nn.BatchNorm2d(outchannels)

        if inchannels != outchannels:
            self.conv1x1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.conv1x1:
            x = self.conv1x1(x)
        out += x
        return F.relu(out)



class MCNN_layer(nn.Module):
    # define the convolution layer in network
    def __init__(self, in_channels, out_channels, ksize, padding, stride):
        super(MCNN_layer,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, ksize, stride, padding)
        self.relu1 = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        return out

class Attention_two_conv(nn.Module):
    def __init__(self, out_channels, ksize, stride, padding):
        super(Attention_two_conv, self).__init__()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, ksize, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, ksize, stride, padding)


    def forward(self, input):
        out_interp1 = self.bn1(input)
        out_interp1 = self.conv1(out_interp1)
        out_interp1 = self.bn2(out_interp1)
        out_interp1 = self.conv2(out_interp1)
        out_softmax = torch.sigmoid(out_interp1)
        return out_softmax

class ResidualBlock1(nn.Module):
    '''
        implement the residual module
        '''

    def __init__(self, inchannels, outchannels, stride=1):
        super(ResidualBlock1, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(outchannels)

        if inchannels != outchannels:
            self.conv1x1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv1x1:
            x = self.conv1x1(x)
        out += x
        return F.relu(out)

class ResidualBlock2(nn.Module):
    '''
        implement the residual module
        '''

    def __init__(self, inchannels, outchannels, stride=1):
        super(ResidualBlock2, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=stride)

        if inchannels != outchannels:
            self.conv1x1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.conv1x1:
            x = self.conv1x1(x)
        out += x
        return F.relu(out)

class ResidualBlock3(nn.Module):
    '''
        implement the residual module
        '''

    def __init__(self, inchannels, outchannels, stride=1):
        super(ResidualBlock3, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=stride)

        if inchannels != outchannels:
            self.conv1x1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.conv1x1:
            x = self.conv1x1(x)
        out += x
        out = self.relu2(out)
        out = self.conv3(out)
        return out

