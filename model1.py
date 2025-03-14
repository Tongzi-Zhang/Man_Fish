import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from linformer import Linformer


class MCNN_new(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, load_weights=False, pool_sizes=[1, 2, 4], use_mixed_precision=True):
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

        # self.self_attention = LinformerAttentionExample(32, num_heads=4, num_classes=190*252)
        self.multi_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)

        self.vgg_part2 = nn.Sequential(nn.Conv2d(64, 20, 3, padding=1, stride=1),
                                       nn.Conv2d(20, 20, 3, padding=1, stride=1),
                                       nn.Conv2d(20, 20, 3, padding=1, stride=1), nn.ReLU(),
                                       ResidualBlock(20, 20))

        self.vgg_part3 = nn.Sequential(nn.Conv2d(20, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1), nn.ReLU(),
                                       ResidualBlock(16, 16))

        # self.vgg_part2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, stride=1),
        #                                nn.Conv2d(32, 32, 3, padding=1, stride=1),
        #                                nn.Conv2d(32, 32, 3, padding=1, stride=1), nn.ReLU(),
        #                                ResidualBlock(32, 32))
        #
        # self.vgg_part3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, stride=1),
        #                                nn.Conv2d(16, 16, 3, padding=1, stride=1),
        #                                nn.Conv2d(16, 16, 3, padding=1, stride=1), nn.ReLU(),
        #                                ResidualBlock(16, 16))
        self.dropout = nn.Dropout(p=0.2)
        self.vgg_part4 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1),
                                       nn.Conv2d(16, 16, 3, padding=1, stride=1), nn.ReLU())
        self.dropout = nn.Dropout(p=0.2)
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
                # 获取x_part1的batch_size, channels, height, width
                batch_size, channels, height, width = x_part1.size()
                #  LinformerAttention ==================================
                # # 需要将特征图展平到适应Linformer的输入格式（batch_size, seq_len, embed_dim）
                # seq_len = height * width  # 图像的序列长度
                # embed_dim = channels  # 特征维度是通道数
                # x_part1 = x_part1.view(batch_size, channels, seq_len).transpose(1, 2)

                # x_part1 = self.self_attention(x_part1)
                # x_part1 = x_part1.reshape(batch_size,channels,height,width)
                # ===============================================================

                # 调整形状，适配多头注意力输入
                x_flat = x_part1.view(batch_size,channels,-1).permute(0, 2, 1)
                # 多头注意力计算
                attn_output, _ = self.multi_attention(x_flat, x_flat, x_flat)
                # 将多头注意力的输出恢复为卷积特征图的形状
                attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
                x_part1 = torch.cat([x_part1, attn_output], dim=1)
                x_part2 = self.vgg_part2(x_part1)

                x_part3 = self.vgg_part3(x_part2)
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

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        # 定义查询、键、值的卷积层
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 自注意力的缩放系数

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # 查询、键、值的卷积
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (batch_size, H*W, C//8)
        key = self.key_conv(x).view(batch_size, -1, H * W)  # (batch_size, C//8, H*W)
        value = self.value_conv(x).view(batch_size, -1, H * W)  # (batch_size, C, H*W)
        # 计算注意力分数
        attention = torch.bmm(query, key)  # (batch_size, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        # 根据注意力权重生成新的特征图
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (batch_size, C, H*W)
        out = out.view(batch_size, C, H, W)
        # 使用gamma进行缩放，并与输入特征图相加
        out = self.gamma * out + x
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)

    def forward(self, x):
        # x 形状为 (seq_len, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x)
        return attn_output




class LinformerAttentionExample(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super(LinformerAttentionExample, self).__init__()
        self.encoder = Linformer(
            dim=embed_dim,
            seq_len=num_classes,  # 假设输入序列的长度是128
            depth=6,
            heads=num_heads,
            k=256  # k是低秩矩阵的维度
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq, channel = x.size()
        x = x.view(batch_size, seq, channel)
        x = self.encoder(x)  # 输入形状 (batch_size, seq_len, embed_dim)
        return x