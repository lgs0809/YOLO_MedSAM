import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplabv3_plus.xception import xception
from deeplabv3_plus.mobilenetv2 import mobilenetv2

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        # Different dilation rates for ASPP
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='xception', pretrained=True, downsample_factor=16):
        super(DeepLabV3Plus, self).__init__()
        
        if backbone == 'xception':
            self.backbone = xception(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == 'mobilenet':
            self.backbone = mobilenetv2(pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))
        
        # ASPP
        self.aspp = ASPP(in_channels, 256)
        
        # Decoder
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.cat_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        self.cls_conv = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        # Backbone
        if hasattr(self.backbone, 'forward'):
            low_level_features, x = self.backbone(x)
        else:
            # For mobilenet
            features = []
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i == 3:  # Low level features at 1/4 resolution
                    low_level_features = x
                features.append(x)
            x = features[-1]
        
        # ASPP
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        
        # Decoder
        low_level_features = self.shortcut_conv(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.cat_conv(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x

def deeplabv3_plus(num_classes=21, backbone='xception', pretrained=True, downsample_factor=16):
    model = DeepLabV3Plus(num_classes, backbone, pretrained, downsample_factor)
    return model

