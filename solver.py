import torch
from torch import nn
from data_loader import *

class Extractor():
    
    # initialize basic info
    def __init__(self, device, net, train_dataset, val_dataset, loss, lr, optimizer, scheduler):
        super(Extractor).__init__()
        self.loss = loss()
        self.device = device
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.n_train, self.n_val = len(self.train_dataset), len(self.val_dataset)
        
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle = True)
        
    def optimize(self, clip = True):
        self.optimizer.zero_grad()
        self.loss.backward()
        if clip:
            nn.utils.clip_grad_value_(net.parameters(), 1)
        self.optimizer.step()
    
    # record parameter change in tensorboard
    def record_para(self, global_step):
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        
    # evaluate the net using validation dataset
    def eval_net(self):
        net.eval()
        
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        n_val = len(loader)
        tot = 0
        for batch in self.val_loader:
            imgs = batch[0].to(self.device, dtype = torch.float32)
            true_masks = batch[1].to(self.device, dtype = torch.float32)

            # no grad traced, speed up
            with torch.no_grad():
                pred_masks = net(imgs)
            pred = torch.sigmoid(pred_masks)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
        net.train()
        return tot / n_val
            return 0
    
    def save_net(self, dir_checkpoint, preffix):
        try:
            os.mkdir(dir_checkpoint)
        except OSError:
            pass
        torch.save(net.state_dict(), dir_checkpoint + f'{preffix}_epoch{epoch + 1}.pth')
        if os.path.exists(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth') & (epoch - 4)//10 != 0:
            os.remove(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth')
    
    # training progress...
    def train(self, epochs, batch_size, save_cp = True, dir_checkpoint = 'checkpoints/', prefix = ''):
        self.batch_size = batch_size
        self.writer = SummaryWriter(comment = f'LR_{self.lr}_BS_{self.batch_size}')
        criterion = nn.BCEWithLogitsLoss()
        global_step = 0
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total = self.n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    imgs = batch[0].to(self.device, dtype = torch.float32)
                    true_masks = batch[1].to(self.device, dtype = torch.float32) # 01
                    pred_masks = net(imgs)
                    loss = self.loss(pred_masks, true_masks)
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    self.optimize(clip = True)

                    # update progress bar
                    pbar.update(imgs.shape[0])
                    global_step += 1

                    # record in tensorboard
                    if global_step % (self.n_train // (10 * batch_size) + 1) == 0:
                        self.record_para()
                        val_score = self.eval_net()
                        scheduler.step(val_score)

                        writer.add_scalar('Dice/test', val_score, global_step)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        writer.add_images('images', imgs, global_step)
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred_0.5', torch.sigmoid(masks_pred) > 0.5, global_step)
            # save the net after each epoch
            if save_cp:
                self.save_net(dir_checkpoint, preffix)
        self.writer.close()
        
        