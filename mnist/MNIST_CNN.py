'''
Author: myzhibei myzhibei@qq.com
Date: 2023-05-17 21:15:42
LastEditors: myzhibei myzhibei@qq.com
LastEditTime: 2023-05-18 16:51:51
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
from torchvision import datasets
from torchvision.transforms import ToTensor

# 文件路径
log_path = r"./logs"  # 日志文件路径
dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./model"  # 模型待存储路径

logf = open(log_path+'/'+'runCNN.log', 'a')
sys.stdout = logf

print(time.time())

CNN_bool = True

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


model = NeuralNetwork().to(device)
print(model)

model_path = "./model/MNIST_model.pth"
if os.path.exists(model_path):
    print(f"Load model {model_path}")
    model.load_state_dict(torch.load(model_path))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN_stack = nn.Sequential(
            # Lambda(preprocess),
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


CNN_model = CNN().to(device)
print(CNN_model)

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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
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


epochs = 20

starttime = time.time()
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
print(f"Time-consuming: {(endtime - starttime)} \n")


# 保存模型
if CNN_bool:
    # model_name = "MNIST_CNN_model" + \
    #     time.strftime("%Y%m%d%H%I%S", time.localtime(time.time()))+".pth"
    model_name = "MNIST_CNN_model_3.pth"
    model_path = model_save_path + '/' + model_name
    torch.save(CNN_model.state_dict(), model_path)
    print(f"Saved PyTorch CNN Model State to {model_path}")

    CNN_model = CNN().to(device)
    CNN_model.load_state_dict(torch.load(model_path))
    print(CNN_model)
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


logf.close()
