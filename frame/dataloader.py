from PIL import Image
import cv2
import os
import random
from frame.config import *


class RoadDataset(Dataset):
    def __init__(self, rootpath, subpath, size = 256, img_transform = None, label_transform = None):
        super(RoadDataset, self).__init__()
        self.img_path = os.path.join(rootpath, subpath, '') # image folder path
        self.label_path = os.path.join(rootpath, subpath + '_labels', '') # label folder path
        self.img_files = list(os.walk(self.img_path))[0][2] # keep order # image file name ending with .tiff
        print("Dataset of path {} initialize. (len: {})".format(os.path.abspath(os.path.join(rootpath, subpath, '')), len(self.img_files)))
        
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ]) if not img_transform else img_transform
        self.label_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ]) if not label_transform else label_transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_files[idx])
        label_path = os.path.join(self.label_path, self.img_files[idx][:-1])
        image = Image.fromarray(cv2.imread(img_path))
        label = Image.fromarray(cv2.imread(label_path))
        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label[0].unsqueeze(0)
    

        