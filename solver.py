import torch
from torch import nn
from data_loader import *

class Extractor():
    def __init__(self, net, device, loss, lr, optimizer, scheduler, scheduler):
        super(Extractor).__init__()
        self.loss = loss()
        self.net = net.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def load_data(root_path, input_size = INPUT_SIZE, output_size = OUTPUT_SIZE):
        self.train_dataset = RoadDataset(road_path, True, INPUT_SIZE, OUTPUT_SIZE)
        self.val_dataset = RoadDataset(road_path, False, INPUT_SIZE, OUTPUT_SIZE)


def train_net(net, device, train_dataset, val_dataset, optimizer, epochs = EPOCH_NUM, lr = LR, save_cp = True, batch_size = BATCH_SIZE):
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
    
    n_val = len(val_dataset)
    n_train = len(train_dataset)