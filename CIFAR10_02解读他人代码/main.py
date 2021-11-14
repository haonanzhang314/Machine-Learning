#本项目特点--采用VGG16实现
# 这是我修改过的，大佬的原代码可以在其主页看
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm   #tqdm是python可扩展的进度条

# 定义超参数start
batch_size = 8  # 批的大小
learning_rate = 1e-2  # 学习率
num_epoches = 5 # 遍历训练集的次数
num_workers =0
momentum = 0.8
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# 定义超参数end

#transform转换
transform = transforms.Compose([
    # transforms.RandomSizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])

# 下载、加载数据集
train_dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)   #num_workers是多线程
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#定义优化器
from VGG16 import VGG16
model = VGG16().to(DEVICE)  #部署到设备上
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format为输出格式，formet括号里的即为左边花括号的输出
    # 训练模型start
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        img, label = img.to(DEVICE), label.to(DEVICE)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)  # 预测最大值所在的位置标签
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
#训练模型end

#测试方法start
    model.eval()  # 模型评估
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:  # 测试模型
        img, label = data
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()
#测试方法end

# 保存模型
torch.save(model.state_dict(), './cnn.pth')

#本文参考：
#https://blog.csdn.net/xiaoheizi_du/article/details/89365916  感谢大佬
#https://github.com/haonanzhang314/Machine-Learning/blob/master/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB.ipynb
