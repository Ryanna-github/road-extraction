> README.md of the current version

# File Structure

- cache/ temporary saved middle data.
- checkpoints/ checkpoints generated while running, some of them would be saved to models/
- experiments/ files used for trials, whose file name usually begin with experiment start-date.
- frame/ the main module.
    - \_\_init\_\_.py
    - evaluate.py: evaluate the model performance.
    - pipeline.py: integration of data processsing, training and testing.
    - utils.py: other handy tools.
    - visualize.py: visualize the results.
- logs/ log data.
- models/ model class definition module.
    - \_\_init\_\_.py
    - unet.py
    - linknet.py
    - ...
- runs/ tensorboard recording files.


---
> README.md of the last version

# 道路提取模型探索

**注意：以下说明在 fakefinal 分支有效，main 正在修改**

> 文件结构
>
> solver.py：模型主要框架，训练，验证，优化器迭代等操作
>
> model.py：用到的模型（Unet，LinkNet34，D-LinkNet34）
>
> loss.py：损失函数
>
> data_loader.py：数据加载文件
>
> tester.py：测试结果
>
> utils.py：工具函数，将不同 shape tensor 转化为 matplotlib.pyplot 可以展示的格式



## 数据

- 训练：[massachusetts-roads-dataset 数据集](https://www.kaggle.com/balraj98/massachusetts-roads-dataset)中表现优良的部分数据，参考 `2021-02-14_data_generator.ipynb`
  - Step1: 根据道路像素点占比 & 图片完整程度筛选数据
  - Step2：切分图片
- 测试：高分一号云南地区图像


## 模型

- 位置：Models 文件夹
  - LinkNet34
  - DLinkNet34
  - FarSegNet
  - DFarSegNet


## Demo

- 参数设置

```{python}
root_path = 'D:/Data/massachusetts-roads-dataset/'
road_path = root_path + "tiff_select2_parts_16/"
INPUT_SIZE, OUTPUT_SIZE = 256, 256
BATCH_SIZE = 4
LR = 0.0005
EPOCH_NUM = 20
```

- 数据载入

```{python}
import data_loader
from torch.utils.data import DataLoader

train_dataset = data_loader.RoadDataset(road_path, INPUT_SIZE, OUTPUT_SIZE, 'train')
val_dataset = data_loader.RoadDataset(road_path, INPUT_SIZE, OUTPUT_SIZE, 'val')

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
```


- 模型训练
> LinkNet34/DLinkNet34 可以使用 solver.py 定义的框架，FarSegNet/DFarSegNet 单独定义对应的训练函数

```{python}
from models import LinkNet34, DLinkNet34, FarSegNet, DFarSegNet
# import ... # 省略

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(params = net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

# for LinkNet34/DLinkNet34
net = model.LinkNet34().to(device)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
sv = Solver(device, net, train_dataset, val_dataset, criterion, LR, BATCH_SIZE, optimizer, scheduler)
sv.train(epochs = EPOCH_NUM, save_cp = True, dir_checkpoint = 'checkpoints/', prefix = 'test')


# for FarSegNet/DFarSegNet
net = FarSegNet().to(device)
writer = SummaryWriter(comment="tensorboard_log{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()))
def update_lr_poly(initial_lr, step, max_step, power):
    return initial_lr * (1-step/max_step)**power

global_step = 0
for epoch in range(EPOCH_NUM):
    net.train()
    
    timer, counter = utils.Timer(), utils.Counter()
    timer.start()
    for step, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        reader_time = timer.elapsed_time()

        loss, miou = net(img, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = float(loss)
        batch_time = timer.elapsed_time()
        counter.append(loss=loss, miou=miou, reader_time=reader_time, batch_time=batch_time)
        eta = utils.calculate_eta(len(train_loader) - step, counter.batch_time)
        print(f"[epoch={epoch + 1}/{EPOCH_NUM}] "
                  f"step={step + 1}/{len(train_loader)} "
                  f"loss={loss:.4f}/{counter.loss:.4f} "
                  f"miou={miou:.4f}/{counter.miou:.4f} "
                  f"batch_time={counter.batch_time:.4f} "
                  f"reader_time={counter.reader_time:.4f} "
                  f"| ETA {eta}",
                  end="\r",
                  flush=True)
        if global_step % 200 == 0:
            writer.add_scalar("Loss", float(loss), global_step=global_step)
            writer.add_scalar("miou", float(miou), global_step=global_step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], global_step = global_step)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                try:
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                except:
                    pass

            net.eval()
            with torch.no_grad():
                probs, preds = net(img, label)
            writer.add_images('image', img, global_step)
            writer.add_images('label/true', label, global_step)
            writer.add_images('label/pred_0.5', preds, global_step)
            net.train()
        global_step += 1
        writer.flush()
        
    print()
    if epoch % 5 != 0:
        if os.path.exists('checkpoints/farseg_epoch{}_global_step{}.pth'.format(epoch-4, global_step)):
            os.remove('checkpoints/farseg_epoch{}_global_step{}.pth'.format(epoch-4, global_step))
    torch.save(net.state_dict(), 'checkpoints/farseg_epoch{}_global_step{}.pth'.format(epoch+1, global_step))
    timer.restart()
    optimizer.param_groups[0]['lr'] = update_lr_poly(LR, epoch, EPOCH_NUM, 0.9)
```

- 结果输出
```{python}
import matplotlib.pyplot as plt

img, lbl = next(iter(train_loader))
net.eval()
tt = net(img.cuda(), lbl.cuda())

# plot prediction
pp = torch.cat([tt[1][0]]*3).permute(1, 2, 0)
plt.imshow((pp*255).cpu())

# plot label
lbl_img = torch.cat([lbl[0]]*3).permute(1, 2, 0)
lbl_img.shape
plt.imshow(lbl_img)
```

- 其他
  - 训练过程可以通过 tensorboard 监控
