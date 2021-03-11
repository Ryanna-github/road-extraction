# 道路提取模型探索

> 现有文件结构
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



## 使用数据

- 训练：massachusetts-roads-dataset 数据集中表现优良的部分数据，将图片切分为 16 部分，最终训练集共 441\*16 张图片，验证集（未经筛选）共 14\*16张图片，测试集（未经筛选）共 49\*16 张图片
- 测试：高分一号云南地区图像，尝试使用样例在文件 `2021-03-07_gf_newdata.ipynb`



## 当前代表性结果

> 由于直接使用 dilated 后的数据预测结果会比真实结果粗很多，故使用 dilated 数据先训练 20 个epoch 后，再在现有模型基础上，用原始数据再进行训练。

- LinkNet 34 + dilated data （20 epochs）

![](predict_result/linknet_dilated_20.png)

- LlinkNet34 + original data （20 epochs）

![](predict_result/linknet_original_20.png)

- D-LinkNet34 + dilated data （20 epochs）

![](predict_result/dlinknet_dilated_20.png)

- D-LlinkNet34 + original data （20 epochs）

  - Still training



## 有效改动

1. 使用 LinkNet 系列，LinkNet 中 Encoder 部分是直接下载好的 ResNet，本就可以有效提取出一些特征
2. 使用 Adam 优化器比 SGD，RMSprop  效果更好
3. D-LinkNet 学习率设置较低才会得到合理结果（0.0002），LinkNet 学习率初始化为 0.001 即可



## 现有问题

1. 现有模型对分布比较均匀的道路提取效果较好，而若图片有大片背景，模型会提取出很多白点，如何添加正则项改进？


2. U-Net 仍然无法得到有效的结果，所有像素点预测结果都是 0，sigmoid 后所有像素点结果都是 0.5，loss 很快就固定在了 0.693 不再发生变化
   1. 怀疑 U-Net 代码出现问题，又找到另外一个版本的 U-Net 代码，但是存在的问题一模一样
   2. 尝试改变初始学习率，0.001，0.,0002 结果都一样（loss 迅速下降并固定在 0.693，所有像素点预测为 0）



## 当前计划

1. 使用 `cv2` 中的函数尝试去除 D-LinkNet 预测结果中的噪声点
2. 使用新数据进行测试，选择有代表性的图片手动打标签后可能会再训练