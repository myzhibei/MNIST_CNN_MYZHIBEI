"""
Author: myzhibei myzhibei@qq.com
Date: 2023-05-04 15:46:59
LastEditors: myzhibei myzhibei@qq.com
LastEditTime: 2023-05-04 16:47:50
FilePath: \实验\手写数字识别\MNIST.py
Description: 在MNIST数据集上进行手写数字识别

Copyright (c) 2023 by myzhibei myzhibei@qq.com, All Rights Reserved. 
"""


import cv2
import torch
import time
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# 设备选择
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 如果有 gpu 那么以 gpu 进行训练 如果没有就使用 cpu 训练

# 文件路径
log_path = r"./logs"  # 日志文件路径
dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./save_model"  # 模型待存储路径

# 参数初始化
epoch = 10  # 训练轮数
filter_epoch = 5  # 滤波次数
batch_size = 64  # 批处理的大小
worker_num = 1  # 同时运行进程数

# 准备数据集
train_data = MNIST(
    dataset_path, train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = MNIST(
    dataset_path,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# 数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)

# 利用 dataloader 加载数据
train_data_loader = DataLoader(
    train_data, batch_size=batch_size, num_workers=worker_num
)
test_data_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    # shuffle=True,
    num_workers=worker_num,
)

# 损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器 设置学习率
learning_rate = 1e-3
# optimizer = torch.optim.SGD(MNIST_CNN().parameters(), lr=learning_rate)

# 添加tensorboard
writer = SummaryWriter(log_path)

# 预处理（同时进行高斯滤波）
transform = transforms.Compose(
    [
        transforms.Resize([28, 28]),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),  # 随机选择的高斯模糊模糊图像
        # transforms.ToTensor(), # 将 PIL Image 或 numpy.ndarray 转换为 tensor ，并归一化到[0,1]之间
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛


# 预训练处理，即高斯滤波处理，相对其他的情况，高斯滤波能达到较好的效果
def PreHandle_Gaussian(dataloader, round):
    i = 0
    stime = time.time()
    # 进行高斯滤波 同时记录当前滤波后的图片情况
    for data in dataloader:
        img, target = data
        transform(img)
        writer.add_images("Pre_Handle_Filter", img, i)
        i += 1
    etime = time.time()
    print(
        "正在进行第{}/{}轮滤波处理(PreHandle_Gaussian方法),花费了{}秒".format(
            round + 1, filter_epoch, etime - stime
        )
    )
    return dataloader


# 进行 filter_epoch 轮滤波处理
def Gaussian_Epoch(dataloader):
    for i in range(filter_epoch):
        dataloader = PreHandle_Gaussian(dataloader, i)
    return dataloader


# 自定义滤波器，实现高斯滤波（使用 OpenCV 库实现的高斯滤波器）
def filter2d(data_loader, kernel_size):  # 自定义滤波器
    for img, target in enumerate(data_loader):  # 对每批图片进行滤波处理
        tmp_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # 读入图片
        dst = cv2.GaussianBlur(tmp_img, (5, 5), sigmaX=1)
    return data_loader


# 保存模型，以字典的形式保存，更加安全
def save_model(model):
    stime = time.time()
    print("------- 正在进行模型保存操作 -------")
    model_name = "MNIST_model_zd.pth"  # 模型名字
    model_path = model_save_path + '/' + model_name  # 模型最终存储路径
    torch.save(model.state_dict(), model_path)
    etime = time.time()
    print(" -------模型已保存，合计花费{}秒 -------".format(etime - stime))
    return model_path


# 加载模型
def load_model(path):
    stime = time.time()
    print("------- 正在进行模型加载操作 -------")
    new_model = MNIST_CNN()
    new_model.load_state_dict(torch.load(path))
    etime = time.time()
    print(" -------模型已加载，合计花费{}秒 -------".format(etime - stime))
    return new_model


# 定义卷积神经网络
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 提取特征，放大特征
        # 注：如果是简单的二分类网络使用 sigmod 会有很好的效果，因为 sigmod 本身区间是0-1具有很好的代表概率作用
        """
            首先对预处理后的数据进行卷积，目的在于将提取图片中的特征,将输出通道数设置为32，目的在于放大特征
            ，便于后续计算，精细化提取特征；
            其次对卷积后的数据进行批标准化，使用下采样保存重要特征信息，同时加快模型收敛速度；
            随后进行一次非线性激活，保证模型的泛化能力。
        """
        self.layer1 = nn.Sequential(  # torch.Size([64, 1, 28, 28])
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=5
            ),  # torch.Size([64, 32, 24, 24])
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )  # torch.Size([64, 32, 24, 24])

        # 提取特征，放大特征
        """
            将输入的特征通道数进一步扩展为64，其余不变；
            最后使用最大池化，进一步保存输入特征，减少训练数据量，提升模型的泛化能力。
        """
        self.layer2 = nn.Sequential(  # torch.Size([64, 32, 24, 24])
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5
            ),  # torch.Size([64, 32, 20, 20])
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=[5, 5],
                stride=1,
                padding=[2, 2],
                dilation=2,
                ceil_mode=False,
            ),
        )  # torch.Size([64, 64, 16, 16])

        # 减少冗余信息
        self.layer3 = nn.Sequential(  # torch.Size([64, 64, 16, 16])
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5
            ),  # torch.Size([64, 64, 12, 12])
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=[5, 5],
                stride=1,
                padding=[2, 2],
                dilation=2,
                ceil_mode=False,
            ),
        )  # torch.Size([64, 64, 8, 8])

        # 减少冗余信息
        self.layer4 = nn.Sequential(  # torch.Size([64, 64, 8, 8])
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3
            ),  # torch.Size([64, 64, 6, 6])
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=[3, 3],
                stride=1,
                padding=[1, 1],
                dilation=2,
                ceil_mode=False,
            ),
        )  # torch.Size([64, 64, 4, 4])

        # 全连接层
        """
            首先将训练出来的结果展平，便于计算；
            其次对其进行一次非线性激活操作，将小于0的元素替换成0，减少非特征元素的干扰；
            随后逐级递减其通道数，最后将其演变为10分类模型。
        """
        self.fc = nn.Sequential(  # torch.Size([64, 64, 4, 4])
            nn.Flatten(),  # torch.Size([64, 64*4*4])
            nn.ReLU(inplace=True),
            nn.Linear(64 * 4 * 4, 256),  # torch.Size([64, 256])
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )  # torch.Size([64, 10])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 初始化
    starttime = time.time()
    model = MNIST_CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # **************** 描绘模型图开始 ****************
    input = torch.ones((64, 1, 28, 28), dtype=torch.float32)  # 测试输入 用于初步检测网络最后的输出形状
    writer.add_graph(model, input)  # 获得网络结构图
    # 查看模型size
    # output = model(input)
    # print(output.shape)

    # **************** 模型训练开始 ****************

    # 进行 filter_epoch 轮滤波处理
    """
        使用高斯模糊对数据集预处理，是平滑图像的方法，通过应用高斯核函数，将像素值进行平滑化处理。
        这个过程可以使图像看起来更加平滑，减少图像中噪声与干扰的影响。
        同时，高斯模糊也可以使得图像中物体与边缘的轮廓变得更加模糊，从而使得一些图像识别算法更易于处理。
    """
    dataloader = Gaussian_Epoch(train_data_loader)  # torch.Size([64, 1, 28, 28])

    # **************** 训练步骤开始 ****************

    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    for i in range(epoch):
        print("------- 第{}/{}轮训练开始 -------".format(i + 1, epoch))

        # 训练步骤开始
        model.train()
        for data in train_data_loader:
            imgs, targets = data
            imgs.to(device)
            targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("运行时间花费：{}".format(end_time - starttime))
                print(
                    "******* 训练次数：{}，Loss：{} *******".format(
                        total_train_step, loss.item()
                    )
                )
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_data_loader:
                imgs, targets = data
                imgs.to(device)
                targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        if i + 1 == epoch:
            save_model(model)

        print("第{}/{}轮测试集上的Loss总和：{}".format(i + 1, epoch, total_test_loss))
        print(
            "第{}/{}轮测试集上的正确率：{}".format(
                i + 1, epoch, total_accuracy / test_data_size * 100
            )
        )
        writer.add_scalar("final_test_loss", total_test_loss, total_test_step)
        writer.add_scalar(
            "final_test_accuracy", total_accuracy / test_data_size, total_test_step
        )
        total_test_step = total_test_step + 1

    endtime = time.time()
    print("模型训练时间{}".format(endtime - starttime))
    writer.close()
