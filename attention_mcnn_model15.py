import torch
from torch import nn

from model.Attention_module import Attention_stage6


class Attention_mcnn(nn.Module):
    def __init__(self, load_weights=False):
        super(Attention_mcnn, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3,16, 9, padding=4),
            nn.ReLU(inplace=True),
            #Attention_stage3(6, 6, 7, 3, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            #Attention_stage3(12, 12, 7, 3, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            Attention_stage6(16, 16, ksize=7, padding=3, stride=1),
            nn.Conv2d(16, 8, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(inplace=True),
            #Attention_stage3(8, 8, 5, 2, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            #Attention_stage3(16, 16, 5, 2, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            Attention_stage6(20, 20, ksize=5, padding=2, stride=1),
            nn.Conv2d(20, 10, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2,stride=1),
            nn.ReLU(inplace=True),
            #Attention_stage3(10, 10, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            #Attention_stage3(20, 20, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            Attention_stage6(24, 24, ksize=3, padding=1, stride=1),
            nn.Conv2d(24, 12, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(30, 1, 1, padding=0)
        )

        if not load_weights:
            self._initialize_weights()

    def forward(self, img_tensor):
        x1 = self.branch1(img_tensor)
        x2 = self.branch2(img_tensor)
        x3 = self.branch3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)
        return x



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

