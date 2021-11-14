#该项目特点
#基于LeNet实现
#将网络模型独立出去单独做了个.py文件
#判断最优模型，并将其保存下来
#对效果有直观的展示
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import copy

NUM_WORKERS = 0    #线程数
EPOCH = 5
BATCH_SIZE = 8
lr = 0.001
momentum = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     #


# Normalize(mean, std) mean：每个通道颜色平均值，这里的平均值为0.5，私人数据集自己计算；std：每个通道颜色标准偏差，(原始数据 - mean) / std 得到归一化后的数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练数据加载
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# 测试数据加载
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#训练函数，得到最优参数 start
def train(model, criterion, optimizer, epochs):
    since = time.time()  #获取当前时间的浮点数

    best_acc = 0.0      # 记录模型测试时的最高准确率
    best_model_wts = copy.deepcopy(model.state_dict())  # 记录模型测试出的最佳参数
    for epoch in range(epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch+1, epochs))   #此处现在的循环次数，以及此任务的循环次数

        # 训练模型
        running_loss = 0.0
        for i, data in enumerate(trainloader):  #i表示训练集每个图片的数字, 和BATCH_SIZE有关
            # enumerate()用于可迭代\可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标。 enumerate(trainloader,0) 这里的0表示索引从0开始，1的话从1开始
            #data中包含图像数据（inputs）和标签（labels）
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播，计算损失
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad() #梯度归零
            loss.backward()       #反向传播
            optimizer.step()     #参数更新优化
            #那么为什么optimizer.step()需要放在每一个batch训练中，而不是epoch训练中，这是因为现在的mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，
            # 因此实际上可以将每一次batch_SIZE看做是一次训练，一次训练更新一次参数空间，因而optimizer.step()放在这里

            running_loss += loss.item()
            #CIFAR_10数据集共有60000张图片，10类，一类有6000张
            # 每1000批图片打印训练数据
            if (i != 0) and (i % 1000 == 0):
                print('step: {:d},  loss: {:.3f}'.format(i, running_loss/1000))
                running_loss = 0.0


        # 每个epoch以测试数据的整体准确率为标准测试一下模型 start
        correct = 0
        total = 0   #总计
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)    #输出最大的可能，取0是每列的最大值，1是每行的最大值

                total += labels.size(0)   #此处为共有多少个标签
                correct += (predicted == labels).sum().item() #累计结果计算正确的次数
        acc = correct / total
        #判断正确率是否有上升，有的话就保存模型
        if acc > best_acc:      # 当前准确率更高时更新
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())  #model.state_dict()是pytorch查看网络参数的方法，

    time_elapsed = time.time() - since   #用于计算用时
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('最高准确率: {}%'.format(100 * best_acc))

    # 返回测试出的最佳模型
    model.load_state_dict(best_model_wts)
    return model
    # 每个epoch以测试数据的整体准确率为标准测试一下模型 end
#训练函数，得到最优参数 end

#定义优化器 start
from net import Net

net = Net()
net.to(DEVICE)

# 使用分类交叉熵 Cross-Entropy 作损失函数，动量SGD做优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum )#momentum方法是为了提高SGD寻优能力，具体就是每次迭代的时候减少学习率的大小。

net = train(net, criterion, optimizer, EPOCH)
# 保存模型参数
torch.save(net.state_dict(), 'net_dict.pt')  #保存整个模型
#定义优化器 end

#测试模型start
# 图像类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
net.load_state_dict(torch.load('net_dict.pt'))  # 加载各层参数
net.to(DEVICE)

# 整体正确率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('整体准确率: {}%'.format(100 * correct / total))

print('=' * 30)

# 每一个类别的正确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('{}的准确率 : {:.2f}%'.format(classes[i], 100 * class_correct[i] / class_total[i]))
#测试模型end

import matplotlib.pyplot as plt
import numpy as np

# 定义一个显示图片的函数
def imshow(img):
    # 输入数据：torch.tensor[c, h, w]
    img = img * 0.5 + 0.5     # 反归一
    npimg = np.transpose(img.numpy(), (1, 2, 0))    # [c, h, w] -> [h, w, c]
    plt.imshow(npimg)
    plt.show()

# 取一批图片
testdata = iter(testloader)
images, labels = testdata.next()
imshow(torchvision.utils.make_grid(images))
print('真实类别: ', ' '.join('{}'.format(classes[labels[j]]) for j in range(labels.size(0))))

# 预测是10个标签的权重，一个类别的权重越大，神经网络越认为它是这个类别，所以输出最高权重的标签。
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('预测结果: ', ' '.join('{}'.format(classes[predicted[j]]) for j in range(labels.size(0))))


#本文来源于https://www.cnblogs.com/Hui4401/p/13495932.html 十分感谢大佬