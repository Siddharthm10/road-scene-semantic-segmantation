import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # Ensure x and g are the same spatial size
        if x.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
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
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
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
        self.attention = AttentionGate(F_g=256, F_l=low_level_in_channels, F_int=low_level_in_channels // 2)
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
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
        low_level_feat = self.attention(low_level_feat, x)
        low_level_feat = self.conv_low(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv_cat(x)
        return self.classifier(x)

class DeepLabV3PlusWithAttention(nn.Module):
    def __init__(self, num_classes, backbone='mobilenetv2', pretrained_backbone=True):
        super(DeepLabV3PlusWithAttention, self).__init__()
        if backbone == 'mobilenetv2':
            mobilenet = models.mobilenet_v2(pretrained=pretrained_backbone)
            self.backbone_low = mobilenet.features[:4]   # low-level features (up to layer 3)
            self.backbone_high = mobilenet.features[4:18]  # high-level features (up to layer 17)
            self.low_level_channels = 24
            self.aspp_in_channels = 320
        else:
            raise NotImplementedError("Backbone {} not implemented".format(backbone))

        self.aspp = ASPP(in_channels=self.aspp_in_channels, out_channels=256, atrous_rates=[6, 12, 18])
        self.decoder = Decoder(low_level_in_channels=self.low_level_channels,
                               low_level_out_channels=48,
                               num_classes=num_classes)

    def forward(self, x):
        x = self.backbone_low(x)
        low_level_feat = x
        x = self.backbone_high(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
