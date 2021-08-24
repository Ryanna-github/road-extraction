from frame.config import *
import frame.evaluate as evaluate
import time

class Solver():
    def __init__(self, net, train, val, loss, lr, batch_size):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr
        
        self.n_train, self.n_val = len(train), len(val)
        self.train_loader = DataLoader(train, self.batch_size, shuffle = True, num_workers = 8)
        self.val_loader = DataLoader(val, self.batch_size, shuffle = True, num_workers = 8)
        
        print("Net: {}".format(type(self.net)))
    
    # evaluate on validation dataset
    def eval_net(self):
        self.net.eval()
        self.eval_criterion = evaluate.DiceLoss()
        tot = 0
        for img, label in self.val_loader:
            img = img.to(self.device, dtype = torch.float32)
            label = label.to(self.device, dtype = torch.float32)
            with torch.no_grad(): # no grad traced, speed up
                pred = (torch.sigmoid(self.net(img)) > 0.5).float()
            tot += self.eval_criterion(pred, label).item()
        self.net.train()
        return tot / self.n_val
        
    # record parameter change in tensorboard
    def record(self, global_step, img, label, pred, cur_loss, val_score):
        # scalar
        self.writer.add_scalar('Loss/train', cur_loss.item(), global_step)
        self.writer.add_scalar('Dice/test', val_score, global_step)
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # images
        self.writer.add_images('images', img, global_step)
        self.writer.add_images('masks/true', label, global_step)
        self.writer.add_images('masks/pred', pred * 255, global_step)
        self.writer.add_images('masks/pred_0.5', torch.sigmoid(pred) > 0.5, global_step)
        
        # tensor
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        
    
    def train(self, optimizer, scheduler, epochs, criterion = nn.BCEWithLogitsLoss(), clip_value = 5):
        self.start_time = time.time()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(comment = f'LR_{self.lr}_BS_{self.batch_size}')
        global_step = 0
        
        # Register a hook to avoid gradient explosion
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        for p in self.net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0
            
            with tqdm(total = self.n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for img, label in self.train_loader:
                    # prediction
                    img = img.to(self.device, dtype = torch.float32)
                    label = label.to(self.device, dtype = torch.float32) # 01
                    pred = self.net(img)
                    
                    # backward propogation
                    cur_loss = self.criterion(pred, label)
                    self.optimizer.zero_grad()
                    cur_loss.backward()
                    epoch_loss += cur_loss.item()
                    
                    # record & evaluate
                    val_score = self.eval_net()
                    self.record(global_step, img, label, pred, cur_loss, val_score)

                    # update progress bar
                    pbar.set_postfix(**{'Loss (batch)': cur_loss.item()})
                    pbar.update(img.shape[0])
                    global_step += 1

                    # record in tensorboard
                    # self.n_train // (10*self.batch_size): the number of 10 batches included in training dataset
                    # global_step % (self.n_train // (10 * self.batch_size) + 1) == 0: every 10*self.batch_size record once
                    if global_step % (self.n_train // (10 * self.batch_size) + 1) == 0:
                        self.record(global_step)
                        val_score = self.eval_net()
            self.scheduler.step() # if ReduceLROnPlateau, val_score is needed as the param
            self.save_net(epoch, dir_checkpoint, prefix)
            
        self.writer.close()
        self.end_time = time.time()
        self.elapse_time = self.end_time - self.start_time
        self.log_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        
    def save_net(self, epoch, dir_checkpoint = 'checkpoints/', prefix = 'test'):
        try:
            os.mkdir(dir_checkpoint)
        except OSError:
            pass
        torch.save(self.net.state_dict(), dir_checkpoint + f'{preffix}_epoch{epoch + 1}.pth')
        if os.path.exists(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth') & (epoch - 4)//10 != 0:
            os.remove(dir_checkpoint + f'{preffix}_epoch{epoch - 4}.pth')

                        