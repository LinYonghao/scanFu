## Faster R-CNN 实现福字检测

## 一.介绍

目前使用`pytorch` 自带的`Faster R-CNN`模型训练的福字检测。

数据集是百度图片下载`LabelImg`手动标记的，大概是100张福字。

后期使用[Nanodet-Plus](https://zhuanlan.zhihu.com/p/306530300) 模型训练One-stage网络，参数量低，可以ncnn部署移动端或嵌入式设备

![](https://s2.loli.net/2022/06/21/qjFYOgPQSmD2RXG.png)

## 二.依赖&环境

###### 开发环境：

- WSL2 Ubuntu distribution

- Windows 11

- CUDA 11.6 (WSL2版)

- 显卡：GTX 1650TI

###### 依赖：

- Pytorch 1.9.0

- albumentations 1.2.0 图片数据增强

- tqdm 4.62.3

- opencv 4.5.5.64

- numpy 1.21.5

## 三.前向推理

**1.下载权重**

**2.安装依赖**

```shell
cd <project-dir>
pip install -r requirements.txt
```

**3.开跑**

```shell
python predict.py -m
```

## 四.训练

**1.数据集准备**

**2.修改配置**

**3.开训**

```shell
python train.py
```

4.visdom 查看训练loss

```python
http://localhost:8097
```



## 参考：

- [一文读懂Faster RCNN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/31426458)

- [[1506.01497] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (arxiv.org)](https://arxiv.org/abs/1506.01497)

- https://www.bilibili.com/video/BV1BK41157Vs

- [TorchVision Faster R-CNN 微调，实战 Kaggle 小麦检测_](https://blog.csdn.net/winorlose2000/article/details/114358777)

- [PyTorch 的可视化利器 Visdom - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/98563580)

**Nanodet相关**

- [YOLO之外的另一选择，手机端97FPS的Anchor-Free目标检测模型NanoDet现已开源~ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/306530300)

- [RangiLyu/nanodet: NanoDet-Plus⚡作者Github实现](https://github.com/RangiLyu/nanodet)
  
  

## TODO

- [x] 春节福字检测

- [ ] 手写福字检测

- [ ] 实现数据集单张图片多次增强

- [ ] NanoDet实现

- [ ] ncnn部署
