from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
import os
import random

classes = ['background', 'road']
colormap = [[0 , 0, 0], [255, 255, 255]]

# 获取给定路径下所有图片路径
def get_file_subpaths(path, whole = True, sort = True):
    path_prefix = path if whole else ""
    subpaths = [path_prefix + file_name for file_name in list(os.walk(path))[0][2]]
    if sort:
        subpaths.sort()
    return subpaths
    
# get train/val/test dataset
class RoadDataset(Dataset):
    def __init__(self, root_path, input_size, output_size, data_class = 'train', sample_ratio = None, seeds = 999):
        super(RoadDataset, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.root_path = root_path
        
        # absolute paths
        self.data_list, self.label_list = self.read_images(type_str = data_class, sample_ratio = sample_ratio, seeds = seeds)
        self.len = len(self.data_list)
        
        # text out
        print('{} dataset set: {} pairs with sample ratio {}'.format(data_class, self.len, sample_ratio))
    
    # get paths (images with corresponding labels)
    def read_images(self, type_str, sample_ratio = None, seeds = 999):
        data = [self.root_path + "/" + type_str + "/" + file_name for file_name in list(os.walk(self.root_path+"/" + type_str + "/"))[0][2]]
        if sample_ratio is not None:
            random.seed(999)
            idx_sample = random.sample(range(len(data)), int(len(data)*sample_ratio))
            data = [data[i] for i in idx_sample]
        file_names_order = [path.split("/")[-1][:-1] for path in data]
        labels = [self.root_path + "/" + type_str + "_labels/" + file_name for file_name in file_names_order]
        return data, labels
    
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img, label = self.change_to_tensor(img, label)
        return img, label
    
    def __len__(self):
        return self.len
    
    # 0-255 to 0-1 in tensors
    def change_to_tensor(self, img, label):
        img = cv2.imread(img)
        label = cv2.imread(label)
        # for every class
        for i, cm in enumerate(colormap):
            label[(label[:, :, 0] == cm[0]) & (label[:, :, 1] == cm[1]) & (label[:, :, 2] == cm[2])] = i * 255
        label = label[:, :, 0] 
        transform_data = transforms.Compose([transforms.Resize([self.input_size, self.input_size], 0), transforms.ToTensor(),])
        transform_label = transforms.Compose([transforms.Resize([self.output_size, self.output_size], 0), transforms.ToTensor(),])
        img_tensor, label_tensor = transform_data(Image.fromarray(img)), transform_label(Image.fromarray(label))
        
        return img_tensor, label_tensor
    
    