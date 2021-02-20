# 道路提取模型探索

## 使用数据

massachusetts-roads-dataset 数据集中表现优良的部分数据，将图片切分为四部分，最终训练集共 92\*4 张图片，验证集（未经筛选）共 14\*4张图片，测试集（未经筛选）共 49\*4 张图片



## 模型尝试

### U-Net

测试数据

![](/predict_result/unet_img.jpg)

![](/predict_result/unet_lbl.jpg)

1. 使用 SGD 结果

   1. 训练 7 个 epoch (每个 epoch 最终 loss 都完全相同，手动停止)
   2. threshold = 0.38 时结果如下

   ![](/predict_result/unet_sgd_038.jpg)

2. 使用 RMSprop 结果

   1. 训练 9 个 epoch（每个 epoch loss 有微小差别，但 tensorboard 中观察结果很差，手动停止）
   2. threshold = 0.29 时结果如下

![](/predict_result/unet_rms_029.jpg)