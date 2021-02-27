import numpy as np
import torch
import matplotlib.pyplot as plt

TEST_SAVE_PATH = '/home/renyan/road-extraction/predict_result/'

class Tester():
    def __init__(self, net, device, dir_stat, dir_checkpoint = 'checkpoints/', save_path = TEST_SAVE_PATH):
        self.device = device
        self.save_path = save_path
        self.net = net.to(self.device)
        self.net.load_state_dict(torch.load(dir_checkpoint + dir_stat))
        
    def test_one(self, img, threshold = 0.5, show = True, save_name = None):
        self.img = img
        self.threshold = threshold
        self.pred = self.net(img.unsqueeze(dim=0).to(self.device))
        self.pred = (torch.sigmoid(self.pred).squeeze(0) > self.threshold) * 255
        self.pred = self.pred.detach().cpu().numpy()
        self.pred = np.concatenate([self.pred]*3).transpose((1, 2, 0))
        if show:
            self.show_one()
        if save_name:
            plt.savefig(self.save_path + save_name)
    
    def show_one(self):
        plt.imshow(self.pred)