'''
Author: myzhibei myzhibei@qq.com
Date: 2023-05-17 21:15:42
LastEditors: myzhibei myzhibei@qq.com
LastEditTime: 2023-05-17 21:48:10
FilePath: \手写数字识别\mnist\MNIST_CNN.py
Description: 

Copyright (c) 2023 by myzhibei myzhibei@qq.com, All Rights Reserved. 
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 文件路径
log_path = r"./logs"  # 日志文件路径
dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./model"  # 模型待存储路径

# 下载数据集
# Download training data from open datasets.
training_data = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root=dataset_path,
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
