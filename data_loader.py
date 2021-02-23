from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
import os


classes = ['background', 'road']
colormap = [[0 , 0, 0], [255, 255, 255]]

# 获取给定路径下所有图片路径
def get_file_subpaths(path, whole = True, sort = True):
    path_prefix = path if whole else ""
    subpaths = [path_prefix + file_name for file_name in list(os.walk(path))[0][2]]
    if sort:
        subpaths.sort()
    return subpaths
    
# 获取图片及其标签路径（图片和标签配对）
def read_images(root_path, train):
    type_str = "train" if train else "val"
    data = [root_path + "/" + type_str + "/" + file_name for file_name in list(os.walk(root_path+"/" + type_str + "/"))[0][2]]
    file_names_order = [path.split("/")[-1][:-1] for path in data]
    labels = [root_path + "/" + type_str + "_labels/" + file_name for file_name in file_names_order]
    return data, labels

# 原图 0-255，tensor 中 0-1
def change_to_tensor(img, label, tensor_size, output_size):
    img = cv2.imread(img)
    label = cv2.imread(label)
    # 对于每一种类别
    for i, cm in enumerate(colormap):
        # 每个通道都符合条件（在本例子中实际只限制一个通道即可）
        label[(label[:, :, 0] == cm[0]) & (label[:, :, 1] == cm[1]) & (label[:, :, 2] == cm[2])] = i * 255
    label = label[:, :, 0] # 只取一个通道结果作为标签
    
    transform_data = transforms.Compose([transforms.Resize([tensor_size, tensor_size], 0),
                                    transforms.ToTensor(),])
    transform_label = transforms.Compose([transforms.Resize([output_size, output_size], 0),
                                    transforms.ToTensor(),])
    img_tensor, label_tensor = transform_data(Image.fromarray(img)), transform_label(Image.fromarray(label))
    return img_tensor, label_tensor

class RoadDataset(Dataset):
    def __init__(self, root_path, train, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(RoadDataset, self).__init__()
        
        self.data_list, self.label_list = read_images(root_path = root_path, train = train)
        self.len = len(self.data_list)
        print('Read '+ str(self.len)+' images')
    
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img, label = change_to_tensor(img, label, self.input_size, self.output_size)
        return img, label
    
    def __len__(self):
        return self.len