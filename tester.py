import numpy as np
import torch
import matplotlib.pyplot as plt

from loss import *

class Tester():
    def __init__(self, net, device, dir_stat, save_path, dir_checkpoint = 'checkpoints/'):
        self.device = device
        self.save_path = save_path
        self.net = net.to(self.device)
        self.net.load_state_dict(torch.load(dir_checkpoint + dir_stat))
        print("Tester with net para in {} is ready".format(dir_stat))
        
    # get one prediction image
    def test_one(self, img, threshold = 0.5, show = True, save_name = None):
        self.img = img
        self.threshold = threshold
        self.pred = self.net(img.to(self.device))
        self.pred = (torch.sigmoid(self.pred).squeeze(0) > self.threshold) * 255
        self.pred = self.pred.detach().cpu().numpy()
        self.pred = np.concatenate([self.pred] * 3).transpose((1, 2, 0))
        if show:
            self.show_one()
        if save_name:
            plt.savefig(self.save_path + save_name)
    
    # show one prediction result
    def show_one(self):
        plt.imshow(self.pred)
    
    def save(self, img, lbl, preffix = 'test_tmp'):
        pass
        
    # get dice & IoU scores for all imgs in test_paths
    def test_score(self, img_paths, lbl_paths):
        self.dice_score = []
        self.iou_score = []
        for i, (img, lbl) in enumerate(zip(img_paths, lbl_paths)):
            pass
        