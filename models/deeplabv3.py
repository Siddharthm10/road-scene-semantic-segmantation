import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3x3 conv branches with different dilation rates
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                      dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                      dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Concatenate and project
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, low_level_in_channels, low_level_out_channels, num_classes):
        super(Decoder, self).__init__()
        # Reduce low-level feature channels
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
        # After concatenation of ASPP output and low-level features
        self.conv_cat = nn.Sequential(
            nn.Conv2d(low_level_out_channels + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_low(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv_cat(x)
        x = self.classifier(x)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained_backbone=True):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained_backbone)
            # Use ResNet layers:
            # conv1, bn1, relu, maxpool, layer1 (low-level), layer2, layer3, layer4 (high-level)
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,   # low-level features (256 channels)
                resnet.layer2,
                resnet.layer3,
                resnet.layer4    # high-level features (2048 channels)
            )
            self.low_level_channels = 256
            self.aspp_in_channels = 2048
        else:
            raise NotImplementedError("Backbone {} not implemented".format(backbone))
        
        # Atrous rates (for output stride 16) are typically [6, 12, 18]
        self.aspp = ASPP(in_channels=self.aspp_in_channels, out_channels=256, atrous_rates=[6, 12, 18])
        # Decoder: reduce low-level channels to 48 and combine with ASPP output
        self.decoder = Decoder(low_level_in_channels=self.low_level_channels,
                               low_level_out_channels=48,
                               num_classes=num_classes)

    def forward(self, x):
        # Manually split backbone features to extract low-level and high-level features
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        low_level_feat = self.backbone[4](x)  # layer1, low-level
        x = self.backbone[5](low_level_feat)    # layer2
        x = self.backbone[6](x)                 # layer3
        high_level_feat = self.backbone[7](x)     # layer4, high-level
        
        x = self.aspp(high_level_feat)
        x = self.decoder(x, low_level_feat)
        # Upsample to the original image size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x
