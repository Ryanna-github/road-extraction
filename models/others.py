import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Function
from torchsummary import summary
import numpy as np
from functools import partial
from torchvision import models


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