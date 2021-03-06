import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Function
from torchsummary import summary
import numpy as np
from functools import partial
from torchvision import models


# ================================= Model 1: U-Net =========================================
class DoubleConv(nn.Module):
    # 输入输出的 feature map 个数可以控制
    # 但是图片具体大小根据图片本身大小决定
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = x2[:,:, (diffY // 2): -(diffY - diffY // 2), (diffY // 2): -(diffX - diffX // 2)]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, bilinear = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels # 图片的通道数（最初输入）
        self.n_classes = n_classes # 最终类别墅（最终输出）
        self.bilinear = bilinear # -- 暂不用

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# ================================= Model 2: LinkNet =========================================
class LinkNet34(nn.Module):
    def __init__(self, num_classes = 1):
        super(LinkNet34, self).__init__()
        self.nonlinearity = partial(F.relu,inplace=True)
        self.n_classes = num_classes

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = self.nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = self.nonlinearity
        self.finalconv3 = nn.Conv2d(32, self.n_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()
        self.nonlinearity = partial(F.relu,inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = self.nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = self.nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = self.nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
    
# ================================= Model 3: D-LinkNet =========================================

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.nonlinearity = partial(F.relu,inplace = True)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = self.nonlinearity(self.dilate1(x))
        dilate2_out = self.nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = self.nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = self.nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DLinkNet34(nn.Module):
    def __init__(self, num_classes = 1):
        super(DLinkNet34, self).__init__()
        self.nonlinearity = partial(F.relu,inplace = True)
        self.n_classes = num_classes

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained = True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = self.nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = self.nonlinearity
        self.finalconv3 = nn.Conv2d(32, self.n_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


# =============================== Unet (Back up) ==================================
class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        
        self.down1 = self.conv_stage(3, 8)
        self.down2 = self.conv_stage(8, 16)
        self.down3 = self.conv_stage(16, 32)
        self.down4 = self.conv_stage(32, 64)
        self.down5 = self.conv_stage(64, 128)
        self.down6 = self.conv_stage(128, 256)
        self.down7 = self.conv_stage(256, 512)
        
        self.center = self.conv_stage(512, 1024)
        #self.center_res = self.resblock(1024)
        
        self.up7 = self.conv_stage(1024, 512)
        self.up6 = self.conv_stage(512, 256)
        self.up5 = self.conv_stage(256, 128)
        self.up4 = self.conv_stage(128, 64)
        self.up3 = self.conv_stage(64, 32)
        self.up2 = self.conv_stage(32, 16)
        self.up1 = self.conv_stage(16, 8)
        
        self.trans7 = self.upsample(1024, 512)
        self.trans6 = self.upsample(512, 256)
        self.trans5 = self.upsample(256, 128)
        self.trans4 = self.upsample(128, 64)
        self.trans3 = self.upsample(64, 32)
        self.trans2 = self.upsample(32, 16)
        self.trans1 = self.upsample(16, 8)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.max_pool = nn.MaxPool2d(2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU()
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )
    
    def forward(self, x):
        conv1_out = self.down1(x)
        conv2_out = self.down2(self.max_pool(conv1_out))
        conv3_out = self.down3(self.max_pool(conv2_out))
        conv4_out = self.down4(self.max_pool(conv3_out))
        conv5_out = self.down5(self.max_pool(conv4_out))
        conv6_out = self.down6(self.max_pool(conv5_out))
        conv7_out = self.down7(self.max_pool(conv6_out))
        
        out = self.center(self.max_pool(conv7_out))
        #out = self.center_res(out)

        out = self.up7(torch.cat((self.trans7(out), conv7_out), 1))
        out = self.up6(torch.cat((self.trans6(out), conv6_out), 1))
        out = self.up5(torch.cat((self.trans5(out), conv5_out), 1))
        out = self.up4(torch.cat((self.trans4(out), conv4_out), 1))
        out = self.up3(torch.cat((self.trans3(out), conv3_out), 1))
        out = self.up2(torch.cat((self.trans2(out), conv2_out), 1))
        out = self.up1(torch.cat((self.trans1(out), conv1_out), 1))

        out = self.conv_last(out)

        return out

# ============================= D-UNet2 ===================================
class Dblock_unet(nn.Module):
    def __init__(self,channel):
        super(Dblock_unet, self).__init__()
        self.nonlinearity = partial(F.relu,inplace=True)
        self.dilate1 = nn.Conv2d(channel//2, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=3, dilation=32, padding=32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = self.nonlinearity(self.dilate1(x))
        dilate2_out = self.nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = self.nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = self.nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = self.nonlinearity(self.dilate5(dilate4_out))
        dilate6_out = self.nonlinearity(self.dilate6(dilate5_out))
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out
        return out
    
class DUNet(nn.Module):
    def __init__(self):
        super(DUNet, self).__init__()
        self.nonlinearity = partial(F.relu,inplace=True)
        
        vgg13 = models.vgg13(pretrained=True)

        self.conv1 = vgg13.features[0]
        self.conv2 = vgg13.features[2]
        self.conv3 = vgg13.features[5]
        self.conv4 = vgg13.features[7]
        self.conv5 = vgg13.features[10]
        self.conv6 = vgg13.features[12]
        
        self.dilate_center = Dblock_unet(512)

        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)
        
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.max_pool = nn.MaxPool2d(2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(inplace=True),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(inplace=True)
        )
    
    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        stage1 = self.nonlinearity(self.conv2(self.nonlinearity(self.conv1(x))))
        stage2 = self.nonlinearity(self.conv4(self.nonlinearity(self.conv3(self.max_pool(stage1)))))
        stage3 = self.nonlinearity(self.conv6(self.nonlinearity(self.conv5(self.max_pool(stage2)))))
        
        out = self.dilate_center(self.max_pool(stage3))
        
        out = self.up3(torch.cat((self.trans3(out), stage3), 1))
        out = self.up2(torch.cat((self.trans2(out), stage2), 1))
        out = self.up1(torch.cat((self.trans1(out), stage1), 1))
        
        out = self.conv_last(out)
        
        return out


