import torch
import torch.nn as nn
import torch.nn.functional as F

# 说明：减少了一格vgg2，效果能好一些，MAE到了140
class MCNN_new(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, load_weights=False, pool_sizes=[1, 2, 4]):
        super(MCNN_new, self).__init__()
        # self.use_mixed_precision = use_mixed_precision
        self.branch1 = nn.Sequential(nn.Conv2d(3, 16, 9, padding=4, stride=1),
                                     nn.BatchNorm2d(16),
                                     nn.MaxPool2d(2), nn.ReLU()
                                     )

        self.branch2 = nn.Sequential(nn.Conv2d(3, 20, 7, padding=3, stride=1),
                                     nn.BatchNorm2d(20),
                                     nn.MaxPool2d(2), nn.ReLU())

        self.branch3 = nn.Sequential(nn.Conv2d(3, 24, 5, padding=2, stride=1),
                                     nn.BatchNorm2d(24),
                                     nn.MaxPool2d(2), nn.ReLU())

        self.fuse = nn.Sequential(nn.Conv2d(60, 48, 3, padding=1, stride=1),
                                  nn.Conv2d(48, 48, 3, padding=1, stride=1),
                                  nn.MaxPool2d(2), nn.ReLU())

        self.vgg_part1 = nn.Sequential(nn.Conv2d(48, 32, 3, padding=1, stride=1),
                                       nn.Conv2d(32, 32, 3, padding=1, stride=1), nn.ReLU(),
                                       ResidualBlock(32, 32))


        # self.vgg_part2 = nn.Sequential(nn.Conv2d(32, 20, 3, padding=1, stride=1),
        #                                nn.Conv2d(20, 20, 3, padding=1, stride=1),
        #                                nn.Conv2d(20, 20, 3, padding=1, stride=1), nn.ReLU(),
        #                                ResidualBlock(20, 20))

        self.vgg_part3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1), nn.ReLU(),
                                       ResidualBlock(16, 16))

        self.vgg_part4 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1), nn.ReLU())

        self.spp = SPPLayer(pool_sizes=pool_sizes)
        # self.fc1 = nn.Linear(48*sum([p * p for p in pool_sizes]), 190*252) # 图像大小

        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(48 * 190 * 252,47880)
        # self.density = nn.Unflatten(1,(1,190,252))
        # self.fc1 = nn.Sequential(nn)
        self.density_generator = nn.Sequential(nn.Conv2d(16 * 4, 64, 1, padding=0, stride=1), nn.ReLU(),
                                               nn.Conv2d(64, 32, 1, padding=0, stride=1), nn.ReLU(),
                                               nn.Conv2d(32, 1, 1, padding=0, stride=1))
        if not load_weights:
            self._initialize_weights()

    def forward(self, im_data):
                x1 = self.branch1(im_data)
                x2 = self.branch2(im_data)
                x3 = self.branch3(im_data)
                x = torch.cat((x1, x2, x3), 1)
                x = self.fuse(x)
                x_part1 = self.vgg_part1(x)
                # x_part1 = self.self_attention(x_part1)
                # x_part2 = se  modelf.vgg_part2(x_part1)

                x_part3 = self.vgg_part3(x_part1)
                x_part4 = self.vgg_part4(x_part3)
                # 增加空间金字塔池化模块
                x_spp = self.spp(x_part4)
                x_part_final = torch.cat([x_part4, x_spp], dim=1)

                density_map = self.density_generator(x_part_final)
                return density_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SPPLayer(nn.Module):
    def __init__(self, pool_sizes):
        """
        初始化SPP层
        Args:
            pool_sizes: 一个列表，表示金字塔不同层的池化尺寸（例如：[1, 2, 4]）
        """
        super(SPPLayer, self).__init__()
        self.pool_sizes = pool_sizes  # 不同尺度的池化窗口大小

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的特征图 (batch_size, channels, height, width)
        Returns:
            拼接后的特征向量
        """
        spp_output = []
        spp_features = []
        batch_size, channels, w, h = x.size()
        spp_output = []
        for pool_size in self.pool_sizes:
            # 自适应平均池化，将特征图池化到指定的尺寸
            pooled_max = F.adaptive_max_pool2d(x, output_size=(pool_size, pool_size))
            #　pooled = pooled.view(batch_size,-1,1,1).expand(-1,-1,h,w)
            pooled_max_resized = F.interpolate(pooled_max, size=(w, h), mode='bilinear', align_corners=False)

            pooled_avg = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
            pooled_avg_resized = F.interpolate(pooled_avg, size=(w, h), mode='bilinear', align_corners=False)

            pooled = (pooled_max_resized + pooled_avg_resized) / 2

            spp_features.append(pooled)
        # 将不同尺度的池化结果拼接起来,
        spp_output = torch.cat(spp_features, dim=1)

        return spp_output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

