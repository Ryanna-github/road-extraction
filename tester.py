import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from loss import *

class Tester():
    def __init__(self, net, device, dir_stat, test_dataset, threshold, save_path, dir_checkpoint = 'checkpoints/'):
        self.device = device
        self.save_path = save_path
        self.net = net.to(self.device)
        self.net.load_state_dict(torch.load(dir_checkpoint + dir_stat))
        self.test_dataset = test_dataset
        self.test_loader = test_loader = DataLoader(self.test_dataset, 1, shuffle = False)
        self.threshold = threshold
        self.n_test = len(test_dataset)
        print("Tester with net para in {} is ready (threshold = {}, {} pairs in test dataset)".format(dir_stat, self.threshold, self.n_test))
    
    def set_threshold(self, new_threshold):
        print("change threshold from {} to {}".format(self.threshold, new_threshold))
        self.threshold = new_threshold
        
    # get one prediction image
    def test_one(self, img, lbl, show = True, combine = False, verbose = False, save_name = None):
        self.img = img
        self.lbl = lbl
        self.pred = self.net(img.to(self.device))
        self.pred = (torch.sigmoid(self.pred).squeeze(0) > self.threshold).type(torch.float32)
        if show:
            self.show_one(combine)
        if verbose:
            print("Dice: {}\nIoU: {}".format(get_dice(self.pred.squeeze(), self.lbl.squeeze()).item(),
                                  get_iou(self.pred.squeeze(), self.lbl.squeeze()).item()))
        if save_name:
            plt.savefig(self.save_path + save_name)
        return self.pred
    
    # show one prediction result
    # ATTENTION: this function can only be used after `test_one`
    def show_one(self, combine):
        self.pred_plot = torch.cat([self.pred]*3).permute(1, 2, 0) * 255
        if not combine:
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(self.img.permute(1, 2, 0))
            plt.subplot(1,3,2)
            plt.imshow(self..permute(1, 2, 0))
            plt.subplot(1,3,3)
            pred = tt.test_one(test_dataset[idx][0].unsqueeze(dim = 0))
            plt.imshow(self.pred_plot)
        else:
            self.pred_plot_combine = self.pred_plot + self.img
            plt.imshow(self.pred_plot_combine)
    
    def save(self, img, lbl, preffix = 'test_tmp'):
        pass
        
    # get dice & IoU scores for all imgs in test dataset
    def test_score(self, save = False):
        self.dice_score = []
        self.iou_score = []
        self.pred_list = []
        for i, (img, lbl) in enumerate(tqdm(iter(self.test_loader))):
            img, lbl = img.to(self.device), lbl.to(self.device)
            pred = self.test_one(img, show = False, save_name = None)
            self.pred_list.append(pred)
            self.iou_score.append(get_iou(pred.squeeze(), lbl.squeeze()).item())
            self.dice_score.append(get_dice(pred.squeeze(), lbl.squeeze()).item())
        if save:
            if not os.path.exists(self.save_path):
                os.makedirs(dirs)
            
        
    
        
        
        
        
        
        
        
        
        