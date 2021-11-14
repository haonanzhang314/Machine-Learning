# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)				# 卷积层：3通道到6通道，卷积5*5
        self.conv2 = nn.Conv2d(6, 16, 5)			# 卷积层：6通道到16通道，卷积5*5

        self.pool = nn.MaxPool2d(2, 2)				# 池化层，在2*2窗口上进行下采样

		# 三个全连接层 ：16*5*5 -> 120 -> 84 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

	# 定义数据流向
    def forward(self, x):
        x = F.relu(self.conv1(x))        # F.relu 是一个常用的激活函数
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)			# 变换数据维度为 1*(16*5*5)，-1表示根据后面推测

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x