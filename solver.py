import torch
from torch import nn
from data_loader import *
from loss import *
from tester import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

class Solver():
    
    # initialize basic info
    def __init__(self, device, net, train_dataset, val_dataset, loss, lr, batch_size, optimizer, scheduler, net_name):
        self.criterion = loss
        self.device = device
        self.net = net.to(self.device)
        self.net_name = net_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.lr = lr
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.n_train, self.n_val = len(self.train_dataset), len(self.val_dataset)
        
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle = True)
        
        print("Net {} initialized.".format(self.net_name))
        
    def optimize(self, pred_masks, true_masks, clip = True):
        cur_loss = self.criterion(pred_masks, true_masks)
        self.optimizer.zero_grad()
        cur_loss.backward()
        if clip:
            nn.utils.clip_grad_value_(self.net.parameters(), 1)
        self.optimizer.step()
        return cur_loss
        
    
    # record parameter change in tensorboard
    def record_para(self, global_step):
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        
    # evaluate the net using validation dataset
    def eval_net(self):
        self.net.eval()
        self.eval_criterion = dice_loss()
        tot = 0
        for batch in self.val_loader:
            imgs = batch[0].to(self.device, dtype = torch.float32)
            true_masks = batch[1].to(self.device, dtype = torch.float32)
            # no grad traced, speed up
            with torch.no_grad():
                pred_masks = self.net(imgs)
            pred = torch.sigmoid(pred_masks)
            pred = (pred > 0.5).float()
            tot += self.eval_criterion(pred, true_masks).item()
        self.net.train()
        return tot / self.n_val
    
    def save_net(self, epoch, dir_checkpoint, preffix):
        try:
            os.mkdir(dir_checkpoint)
        except OSError:
            pass
        torch.save(self.net.state_dict(), dir_checkpoint + f'{preffix}_epoch{epoch + 1}.pth')
        if os.path.exists(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth') & (epoch - 4)//10 != 0:
            os.remove(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth')
            
    def write_info(self, message = '', file_path = 'train_info.txt'):
        doc = open(file_path, 'a+')
        print(message, file = doc)
        doc.close()
    
    # training progress...
    def train(self, epochs, save_cp = True, dir_checkpoint = 'checkpoints/', prefix = ''):
        self.start_time = time.time()
        self.writer = SummaryWriter(comment = f'LR_{self.lr}_BS_{self.batch_size}')
        global_step = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0
            with tqdm(total = self.n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    imgs = batch[0].to(self.device, dtype = torch.float32)
                    true_masks = batch[1].to(self.device, dtype = torch.float32) # 01
                    pred_masks = self.net(imgs)
                    cur_loss = self.optimize(pred_masks, true_masks, clip = True)
                    
                    epoch_loss += cur_loss.item()
                    
                    # record
                    self.writer.add_scalar('Loss/train', cur_loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': cur_loss.item()})
                    
                    # update progress bar
                    pbar.update(imgs.shape[0])
                    global_step += 1

                    # record in tensorboard
                    if global_step % (self.n_train // (10 * self.batch_size) + 1) == 0:
                        self.record_para(global_step)
                        val_score = self.eval_net()
                        self.scheduler.step(val_score)

                        self.writer.add_scalar('Dice/test', val_score, global_step)
                        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
                        self.writer.add_images('images', imgs, global_step)
                        self.writer.add_images('masks/true', true_masks, global_step)
                        self.writer.add_images('masks/pred_0.5', torch.sigmoid(pred_masks) > 0.5, global_step)
            # save the net after each epoch
            if save_cp:
                self.save_net(epoch, dir_checkpoint, prefix)
        self.writer.close()
        self.end_time = time.time()
        self.elapse_time = self.end_time - self.start_time
        self.log_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.write_info("Infomation recorded in {} of {}\n\ttime used: {}\
                    \n\tepoch number: {}\n\tcurrent lr: {}\n".format(self.log_time, self.net_name, self.elapse_time, epochs, self.lr))
    