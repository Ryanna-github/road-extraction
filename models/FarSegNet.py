import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Function
from torchsummary import summary
import numpy as np
from functools import partial
from torchvision import models

class Decoder(nn.Module):
    def __init__(self, c_in, scale):
        super(Decoder, self).__init__()
        assert scale in [1, 2, 4, 8]
        if scale >= 1:
            self.conv1 = Conv2dBN(c_in, c_in, 3, padding=1)
        if scale >= 4:
            self.conv2 = Conv2dBN(c_in, c_in, 3, padding=1)
        if scale >= 8:
            self.conv3 = Conv2dBN(c_in, c_in, 3, padding=1)
        self.scale = scale

    def forward(self, x):
        if self.scale >= 1:
            x = self.conv1(x)
            if self.scale == 1:
                return x
        if self.scale >= 2:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if self.scale >= 4:
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if self.scale >= 8:
            x = self.conv3(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x
    
class FSModule(nn.Module):
    def __init__(self, cv, cu):
        super(FSModule, self).__init__()

        self.conv1 = Conv2dBN(cv, cu, 1)
        self.conv2 = Conv2dBN(cv, cu, 1)

    def forward(self, v, u):
        x = self.conv1(v)
        r = torch.mul(x, u)
        k = self.conv2(v)
        z = k / (1 + torch.exp(-r))
        return z

class Conv2dBN(nn.Module):
    def __init__(self, c_in, c_out, filter_size, stride=1, padding=0, **kwargs):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, filter_size, stride=stride, padding=padding, **kwargs)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FarSegNet(nn.Module):
    def __init__(self, num_classes = 1, num_feature = 256, pretrained = False, ignore_index = 500):
        super(FarSegNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_feature = num_feature
        self.ignore_index = ignore_index # ignore losses of this class
        self.EPS = 1e-5
        self.current_step = 0
        self.annealing_step = 10000
        self.focal_factor = 2
        self.focal_z = 1.0
        
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone_layer_c2 = nn.Sequential(*list(self.backbone.children())[:5])
        self.backbone_layer_c3 = list(self.backbone.children())[5]
        self.backbone_layer_c4 = list(self.backbone.children())[6]
        self.backbone_layer_c5 = list(self.backbone.children())[7]
        
        self.conv_c6 = nn.Conv2d(2048, num_feature, 1)
        self.conv_c5 = nn.Conv2d(2048, num_feature, 1)
        self.conv_c4 = nn.Conv2d(1024, num_feature, 1)
        self.conv_c3 = nn.Conv2d(512, num_feature, 1)
        self.conv_c2 = nn.Conv2d(256, num_feature, 1)

        self.fs5 = FSModule(num_feature, num_feature)
        self.fs4 = FSModule(num_feature, num_feature)
        self.fs3 = FSModule(num_feature, num_feature)
        self.fs2 = FSModule(num_feature, num_feature)

        self.up5 = Decoder(num_feature, 8)
        self.up4 = Decoder(num_feature, 4)
        self.up3 = Decoder(num_feature, 2)
        self.up2 = Decoder(num_feature, 1)

        self.classify = nn.Conv2d(num_feature, num_classes, 3, padding=1)
    
    def forward(self, x, label = None):
        c2 = self.backbone_layer_c2(x)
        c3 = self.backbone_layer_c3(c2)
        c4 = self.backbone_layer_c4(c3)
        c5 = self.backbone_layer_c5(c4)
        c6 = F.adaptive_avg_pool2d(c5, (1, 1))
        u = self.conv_c6(c6)
        
        p5 = self.conv_c5(c5)
        p4 = (self.conv_c4(c4) + F.interpolate(p5, scale_factor = 2)) / 2.
        p3 = (self.conv_c3(c3) + F.interpolate(p4, scale_factor = 2)) / 2.
        p2 = (self.conv_c2(c2) + F.interpolate(p3, scale_factor = 2)) / 2.
        
        z5 = self.fs5(p5, u)
        z4 = self.fs4(p4, u)
        z3 = self.fs3(p3, u)
        z2 = self.fs2(p2, u)

        o5 = self.up5(z5)
        o4 = self.up4(z4)
        o3 = self.up3(z3)
        o2 = self.up2(z2)
        
        x = (o5 + o4 + o3 + o2) / 4.
        x = F.interpolate(x, scale_factor = 4, mode = "bilinear", align_corners = True)
        logit = self.classify(x)
        
        if self.training:
            return self._get_loss(logit, label), self._get_miou(logit, label)
        else:
            probs = torch.sigmoid(logit)
            preds = (probs > 0.5).int()
            return probs, preds
        
    def _get_loss(self, logit, label):

        logit, label = logit.flatten(), label.flatten()
        loss = nn.BCEWithLogitsLoss(reduction='none')(logit, label)

        probs = torch.sigmoid(logit) # probability
        p = ((1-label) + (-1)**(1+label)*probs).squeeze() # label should be 0 or 1

        z = torch.pow(1.0 - p, self.focal_factor)
        z = self.focal_z * z

        if self.current_step < self.annealing_step:
            z = z + (1 - z) * (1 - self.current_step / self.annealing_step)
        self.current_step += 1

        loss = z * loss
        avg_loss = torch.mean(loss)
        return avg_loss

    def _get_miou(self, logit, label):
        
        miou_sum, miou_count = 0, 0
        for batch_idx in range(label.shape[0]):
            pred_one = (torch.sigmoid(logit[batch_idx].flatten()) > 0.5).int()
            label_one = label[batch_idx].flatten()
            intersection = (pred_one * label_one).sum()
            union = pred_one.sum() + label_one.sum()
            
            miou_sum += intersection / union
            miou_count += 1
            
        return miou_sum / miou_count