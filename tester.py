import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2

from loss import *
from utils import *


# self.img (4-d tensor)
# self.lbl (4-d tensor)
# self.pred (4-d tensor)

class Tester():
    def __init__(self, net, device, dir_stat, test_dataset, threshold, save_path, dir_checkpoint = 'checkpoints/', log_path = 'test_log'):
        self.device = device
        self.save_path = save_path
        self.net = net.to(self.device)
        self.net.load_state_dict(torch.load(dir_checkpoint + dir_stat))
        self.test_dataset = test_dataset
        self.test_loader = test_loader = DataLoader(self.test_dataset, 1, shuffle = False)
        self.threshold = threshold
        self.n_test = len(test_dataset)
        self.log_path = log_path
        print("Tester with net para in {} is ready \n(threshold = {}, {} pairs in test dataset)".format(dir_stat, self.threshold, self.n_test))
    
    # change threshold
    def set_threshold(self, new_threshold):
        print("change threshold from {} to {}".format(self.threshold, new_threshold))
        self.threshold = new_threshold
        
    # get one prediction image
    def test_one(self, img, lbl, show = True, combine = False, verbose = False, save_name = None):
        self.img = img.to(self.device)
        self.lbl = lbl.to(self.device)
        self.pred = self.net(self.img.to(self.device))
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
    def show_one(self, combine = False):
        if not combine:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(change_tensor_to_plot(self.img))
            plt.subplot(1, 3, 2)
            plt.imshow(change_tensor_to_plot(self.lbl))
            plt.subplot(1, 3, 3)
            plt.imshow(change_tensor_to_plot(self.pred))
        else:
            img_pred_out = torch.mul(change_tensor_to_plot(self.img, to_numpy = False), 
                                     1-change_tensor_to_plot(self.pred, to_numpy = False))
            self.combine = torch.add(img_pred_out.cuda(), change_tensor_to_plot(self.pred, to_numpy = False))
            plt.imshow(change_tensor_to_plot(self.combine))
    
    def show_idx(self, idx, combine = False):
        self.img = self.test_dataset[idx][0].unsqueeze(dim = 0).to(self.device)
        self.lbl = self.test_dataset[idx][1].unsqueeze(dim = 0).to(self.device)
        self.pred = self.net(self.img.to(self.device))
        self.pred = (torch.sigmoid(self.pred).squeeze(0) > self.threshold).type(torch.float32)
        self.show_one(combine)

    
    def save(self, subfolder = 'test_result/'):
        if not os.path.exists(self.save_path + subfolder):
            os.makedirs(self.save_path + subfolder)
            print("Build folder of {}".format(self.save_path + subfolder))
        for i, pred in enumerate(tqdm(self.pred)):
            each_file_name = self.test_dataset.datalist[i].split("/")[-1].split(".")[0] + '.png'
            full_file_path = self.save_path + subfolder + each_file_name
            cv2.imwrite(full_file_path, pred*255)
            
        
    # get dice & IoU scores for all imgs in test dataset
    def test_score(self, save = False, verbose = True):
        self.dice_score = []
        self.iou_score = []
        self.pred_list = []
        for i, (img, lbl) in enumerate(tqdm(iter(self.test_loader))):
            img, lbl = img.to(self.device), lbl.to(self.device)
            pred = self.test_one(img, lbl, show = False, save_name = None)
            self.pred_list.append(pred)
            self.iou_score.append(get_iou(pred.squeeze(), lbl.squeeze()).item())
            self.dice_score.append(get_dice(pred.squeeze(), lbl.squeeze()).item())
        if verbose:
            print("current threshold: {}\nmean dice: {}\nmean iou: {}".format(self.threshold, np.mean(self.dice_score), np.mean(self.iou_score)))
        if save:
            self.save()
            
        
    def get_threshold_best(self, lowest = 0.45, highest = 0.55, stride = 0.01):
        candidates = list(np.arange(lowest, highest, stride))
        c_dice_score, c_iou_score = [], []
        for t in candidates:
            self.set_threshold(t)
            self.test_score(save = False, verbose = True)
            c_dice_score.append(np.mean(self.dice_score))
            c_iou_score.append(np.mean(self.iou_score))
        self.threshold_table = pd.DataFrame({"threshold": candidates,
                                 "dice": c_dice_score,
                                 "iou": c_iou_score})
        
        
        
        
        
        
        
        
        