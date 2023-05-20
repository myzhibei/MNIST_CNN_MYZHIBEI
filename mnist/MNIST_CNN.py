'''
Author: myzhibei myzhibei@qq.com
Date: 2023-05-17 21:15:42
LastEditors: myzhibei myzhibei@qq.com
LastEditTime: 2023-05-20 17:45:55
FilePath: \手写数字识别\mnist\MNIST_CNN.py
Description: 

Copyright (c) 2023 by myzhibei myzhibei@qq.com, All Rights Reserved. 
'''
import sys
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# Image datasets and image manipulation
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np


# 文件路径
log_path = r"D:\Log"  # 日志文件路径
dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./model"  # 模型待存储路径

logf = open('./logs/runCNN.log', 'a')
# sys.stdout = logf

print(time.time())

CNN_bool = True

# Gather datasets and prepare them for consumption
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
# 归一化数据


# 下载数据集
# Download training data from open datasets.
training_data = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=True,
    transform=transform,
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root=dataset_path,
    train=False,
    download=True,
    transform=transform,
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Extract a batch of 64 images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print("show image")

# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above

writer = SummaryWriter(log_path)

# Write image data to TensorBoard log dir
writer.add_image('MNIST Images', img_grid)
writer.flush()


# To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if not CNN_bool:
    model = NeuralNetwork().to(device)
    print(model)
    input = torch.ones((64, 1, 28, 28), dtype=torch.float32).to(
        device)  # 测试输入 用于初步检测网络最后的输出形状
    writer.add_graph(model, input)  # 获得网络结构图
    model_path = "./model/MNIST_model.pth"
    if os.path.exists(model_path):
        print(f"Load model {model_path}")
        model.load_state_dict(torch.load(model_path))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.CNN_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


if CNN_bool:
    CNN_model = CNN().to(device)
    print(CNN_model)
    input = torch.ones((64, 1, 28, 28), dtype=torch.float32).to(
        device)  # 测试输入 用于初步检测网络最后的输出形状
    writer.add_graph(CNN_model, input)  # 获得网络结构图
    writer.flush()

    # model_path = "./model/MNIST_CNN_model.pth"
    # if os.path.exists(model_path):
    #     print(f"Load model {model_path}")
    #     CNN_model.load_state_dict(torch.load(model_path))

loss_fn = nn.CrossEntropyLoss()
if CNN_bool:
    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=1e-3, momentum=0.9)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    global total_train_step
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_step = total_train_step + 1
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"train_step:{total_train_step} \t loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalars('Training Loss', {
                               'Training': loss}, total_train_step)


def test(dataloader, model, loss_fn):
    global total_test_step
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print(f"num_batches = {num_batches}")
    model.eval()
    test_loss, correct = 0, 0
    total_test_step = total_test_step + 1
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalars('Validation Loss', {
        'Validation': test_loss}, total_test_step)
    writer.add_scalar("Validation accuracy", correct, total_test_step)


epochs = 25

starttime = time.time()

# 记录训练的次数
global total_train_step
total_train_step = 0
# 记录测试的次数
global total_test_step
total_test_step = 0

if CNN_bool:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, CNN_model, loss_fn, optimizer)
        test(test_dataloader, CNN_model, loss_fn)
else:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
endtime = time.time()
print("Done!")
writer.flush()
writer.close()
print(f"Time-consuming: {(endtime - starttime)} \n")


# 保存模型
if CNN_bool:
    # model_name = "MNIST_CNN_model" + \
    #     time.strftime("%Y%m%d%H%I%S", time.localtime(time.time()))+".pth"
    model_name = "MNIST_CNN_model.pth"
    model_path = model_save_path + '/' + model_name
    torch.save(CNN_model.state_dict(), model_path)
    print(f"Saved PyTorch CNN Model State to {model_path}")
    CNN_model = CNN().to(device)
    CNN_model.load_state_dict(torch.load(model_path))
    # print(CNN_model)

    CNN_model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.unsqueeze(0)
        x = x.to(device)
        pred = CNN_model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
else:
    model_name = "MNIST_model.pth"
    model_path = model_save_path + '/' + model_name

    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
print('Finished')
logf.close()
