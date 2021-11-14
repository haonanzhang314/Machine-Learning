# test01——87%

### VGG

### 图像扩增

### batch_size = 32 

### learning_rate = 0.001 

### num_epoches = 50 

### num_workers =4 

### momentum = 0.8 

1. ```python
   ************************* epoch 1 *************************
   1563it [01:48, 14.45it/s]
   Finish 1 epoch, Loss: 1.600895, Acc: 0.404000
   Test Loss: 1.296623, Acc: 0.533500
   
   ************************* epoch 2 *************************
   1563it [01:48, 14.45it/s]
   Finish 2 epoch, Loss: 1.167767, Acc: 0.583400
   Test Loss: 1.025729, Acc: 0.636400
   
   ************************* epoch 3 *************************
   1563it [01:47, 14.49it/s]
   Finish 3 epoch, Loss: 0.973750, Acc: 0.659900
   Test Loss: 0.993342, Acc: 0.646900
   
   ************************* epoch 4 *************************
   1563it [01:47, 14.54it/s]
   Finish 4 epoch, Loss: 0.857208, Acc: 0.703640
   Test Loss: 0.807573, Acc: 0.718200
   
   ************************* epoch 5 *************************
   1563it [01:47, 14.58it/s]
   Finish 5 epoch, Loss: 0.766039, Acc: 0.737280
   Test Loss: 0.767924, Acc: 0.732100
   
   ************************* epoch 6 *************************
   1563it [01:47, 14.54it/s]
   Finish 6 epoch, Loss: 0.701275, Acc: 0.759820
   Test Loss: 0.713432, Acc: 0.761600
   
   ************************* epoch 7 *************************
   1563it [01:47, 14.54it/s]
   Finish 7 epoch, Loss: 0.646193, Acc: 0.780000
   Test Loss: 0.663220, Acc: 0.772900
   
   ************************* epoch 8 *************************
   1563it [01:47, 14.54it/s]
   Finish 8 epoch, Loss: 0.605146, Acc: 0.791560
   Test Loss: 0.624093, Acc: 0.784300
   
   ************************* epoch 9 *************************
   1563it [01:47, 14.54it/s]
   Finish 9 epoch, Loss: 0.569875, Acc: 0.806760
   Test Loss: 0.617746, Acc: 0.789900
   
   ************************* epoch 10 *************************
   1563it [01:47, 14.55it/s]
   Finish 10 epoch, Loss: 0.541259, Acc: 0.813180
   Test Loss: 0.562139, Acc: 0.810200
   
   ************************* epoch 11 *************************
   1563it [01:47, 14.55it/s]
   Finish 11 epoch, Loss: 0.515041, Acc: 0.824120
   Test Loss: 0.584683, Acc: 0.801700
   
   ************************* epoch 12 *************************
   1563it [01:47, 14.56it/s]
   Finish 12 epoch, Loss: 0.486848, Acc: 0.832500
   Test Loss: 0.543341, Acc: 0.814700
   
   ************************* epoch 13 *************************
   1563it [01:47, 14.55it/s]
   Finish 13 epoch, Loss: 0.462327, Acc: 0.841260
   Test Loss: 0.527300, Acc: 0.820900
   
   ************************* epoch 14 *************************
   1563it [01:47, 14.55it/s]
   Finish 14 epoch, Loss: 0.442085, Acc: 0.849200
   Test Loss: 0.506156, Acc: 0.826900
   
   ************************* epoch 15 *************************
   1563it [01:47, 14.55it/s]
   Finish 15 epoch, Loss: 0.426711, Acc: 0.855220
   Test Loss: 0.494245, Acc: 0.830300
   
   ************************* epoch 16 *************************
   1563it [01:47, 14.55it/s]
   Finish 16 epoch, Loss: 0.408453, Acc: 0.859540
   Test Loss: 0.506342, Acc: 0.832000
   
   ************************* epoch 17 *************************
   1563it [01:47, 14.55it/s]
   Finish 17 epoch, Loss: 0.387428, Acc: 0.867980
   Test Loss: 0.481042, Acc: 0.841800
   
   ************************* epoch 18 *************************
   1563it [01:47, 14.54it/s]
   Finish 18 epoch, Loss: 0.380014, Acc: 0.869240
   Test Loss: 0.456107, Acc: 0.846900
   
   ************************* epoch 19 *************************
   1563it [01:47, 14.55it/s]
   Finish 19 epoch, Loss: 0.362714, Acc: 0.876180
   Test Loss: 0.472405, Acc: 0.839900
   
   ************************* epoch 20 *************************
   1563it [01:47, 14.55it/s]
   Finish 20 epoch, Loss: 0.346784, Acc: 0.882420
   Test Loss: 0.475072, Acc: 0.842700
   
   ************************* epoch 21 *************************
   1563it [01:47, 14.54it/s]
   Finish 21 epoch, Loss: 0.336204, Acc: 0.884760
   Test Loss: 0.495061, Acc: 0.835500
   
   ************************* epoch 22 *************************
   1563it [01:47, 14.54it/s]
   Finish 22 epoch, Loss: 0.328043, Acc: 0.887720
   Test Loss: 0.449675, Acc: 0.849600
   
   ************************* epoch 23 *************************
   1563it [01:47, 14.55it/s]
   Finish 23 epoch, Loss: 0.316723, Acc: 0.893300
   Test Loss: 0.451836, Acc: 0.850100
   
   ************************* epoch 24 *************************
   1563it [01:46, 14.61it/s]
   Finish 24 epoch, Loss: 0.303096, Acc: 0.895900
   Test Loss: 0.447426, Acc: 0.849500
   
   ************************* epoch 25 *************************
   1563it [01:47, 14.60it/s]
   Finish 25 epoch, Loss: 0.295367, Acc: 0.898360
   Test Loss: 0.439646, Acc: 0.857700
   
   ************************* epoch 26 *************************
   1563it [01:47, 14.59it/s]
   Finish 26 epoch, Loss: 0.284602, Acc: 0.904120
   Test Loss: 0.426826, Acc: 0.861300
   
   ************************* epoch 27 *************************
   1563it [01:47, 14.59it/s]
   Finish 27 epoch, Loss: 0.276745, Acc: 0.904820
   Test Loss: 0.419403, Acc: 0.860700
   
   ************************* epoch 28 *************************
   1563it [01:47, 14.56it/s]
   Finish 28 epoch, Loss: 0.271996, Acc: 0.906460
   Test Loss: 0.428001, Acc: 0.858000
   
   ************************* epoch 29 *************************
   1563it [01:47, 14.56it/s]
   Finish 29 epoch, Loss: 0.258404, Acc: 0.911040
   Test Loss: 0.422789, Acc: 0.859400
   
   ************************* epoch 30 *************************
   1563it [01:47, 14.59it/s]
   Finish 30 epoch, Loss: 0.255740, Acc: 0.914140
   Test Loss: 0.424502, Acc: 0.861600
   
   ************************* epoch 31 *************************
   1563it [01:47, 14.59it/s]
   Finish 31 epoch, Loss: 0.241917, Acc: 0.915860
   Test Loss: 0.424551, Acc: 0.864000
   
   ************************* epoch 32 *************************
   1563it [01:47, 14.59it/s]
   Finish 32 epoch, Loss: 0.240587, Acc: 0.917540
   Test Loss: 0.451442, Acc: 0.857800
   
   ************************* epoch 33 *************************
   1563it [01:47, 14.60it/s]
   Finish 33 epoch, Loss: 0.234878, Acc: 0.919300
   Test Loss: 0.424345, Acc: 0.862500
   
   ************************* epoch 34 *************************
   1563it [01:47, 14.59it/s]
   Finish 34 epoch, Loss: 0.223610, Acc: 0.921940
   Test Loss: 0.424382, Acc: 0.864900
   
   ************************* epoch 35 *************************
   1563it [01:47, 14.57it/s]
   Finish 35 epoch, Loss: 0.218907, Acc: 0.924500
   Test Loss: 0.434907, Acc: 0.863200
   
   ************************* epoch 36 *************************
   1563it [01:47, 14.55it/s]
   Finish 36 epoch, Loss: 0.213656, Acc: 0.926160
   Test Loss: 0.393337, Acc: 0.870400
   
   ************************* epoch 37 *************************
   1563it [01:47, 14.59it/s]
   Finish 37 epoch, Loss: 0.211265, Acc: 0.927820
   Test Loss: 0.408507, Acc: 0.869400
   
   ************************* epoch 38 *************************
   1563it [01:47, 14.57it/s]
   Finish 38 epoch, Loss: 0.204425, Acc: 0.930040
   Test Loss: 0.449373, Acc: 0.859000
   
   ************************* epoch 39 *************************
   1563it [01:47, 14.54it/s]
   Finish 39 epoch, Loss: 0.199104, Acc: 0.932420
   Test Loss: 0.459263, Acc: 0.856400
   
   ************************* epoch 40 *************************
   1563it [01:47, 14.54it/s]
   Finish 40 epoch, Loss: 0.191695, Acc: 0.933900
   Test Loss: 0.421711, Acc: 0.864600
   
   ************************* epoch 41 *************************
   1563it [01:47, 14.54it/s]
   Finish 41 epoch, Loss: 0.186800, Acc: 0.935760
   Test Loss: 0.418513, Acc: 0.868300
   
   ************************* epoch 42 *************************
   1563it [01:47, 14.57it/s]
   Finish 42 epoch, Loss: 0.184709, Acc: 0.935820
   Test Loss: 0.411686, Acc: 0.870500
   
   ************************* epoch 43 *************************
   1563it [01:47, 14.59it/s]
   Finish 43 epoch, Loss: 0.179734, Acc: 0.937000
   Test Loss: 0.393837, Acc: 0.874300
   
   ************************* epoch 44 *************************
   1563it [01:47, 14.59it/s]
   Finish 44 epoch, Loss: 0.172526, Acc: 0.940500
   Test Loss: 0.382627, Acc: 0.875700
   
   ************************* epoch 45 *************************
   1563it [01:47, 14.59it/s]
   Finish 45 epoch, Loss: 0.172928, Acc: 0.940480
   Test Loss: 0.402739, Acc: 0.870800
   
   ************************* epoch 46 *************************
   1563it [01:47, 14.59it/s]
   Finish 46 epoch, Loss: 0.168521, Acc: 0.942180
   Test Loss: 0.404802, Acc: 0.876800
   
   ************************* epoch 47 *************************
   1563it [01:47, 14.59it/s]
   Finish 47 epoch, Loss: 0.164091, Acc: 0.945500
   Test Loss: 0.436241, Acc: 0.866200
   
   ************************* epoch 48 *************************
   1563it [01:47, 14.60it/s]
   Finish 48 epoch, Loss: 0.160762, Acc: 0.944040
   Test Loss: 0.391431, Acc: 0.876900
   
   ************************* epoch 49 *************************
   1563it [01:47, 14.59it/s]
   Finish 49 epoch, Loss: 0.156629, Acc: 0.945500
   Test Loss: 0.412474, Acc: 0.873600
   
   ************************* epoch 50 *************************
   1563it [01:47, 14.59it/s]
   Finish 50 epoch, Loss: 0.151955, Acc: 0.948600
   Test Loss: 0.417550, Acc: 0.870500
   ```




# test02--85%

### batch_size = 8 

### learning_rate = 0.005 

### num_epoches = 100 

### num_workers =8 

### momentum = 0.8 

```python
250it [05:27, 19.08it/s]
Finish 1 epoch, Loss: 1.631940, Acc: 0.402380
Test Loss: 1.229264, Acc: 0.561700
6250it [05:27, 19.08it/s]
Finish 2 epoch, Loss: 1.100300, Acc: 0.622700
Test Loss: 0.872377, Acc: 0.709900
6250it [05:27, 19.08it/s]
Finish 3 epoch, Loss: 0.861430, Acc: 0.716740
Test Loss: 0.793398, Acc: 0.742900
6250it [05:27, 19.07it/s]
Finish 4 epoch, Loss: 0.695563, Acc: 0.771320
Test Loss: 0.784753, Acc: 0.744900
6250it [05:27, 19.08it/s]
Finish 5 epoch, Loss: 0.577089, Acc: 0.810820
Test Loss: 0.598185, Acc: 0.807100
6250it [05:27, 19.07it/s]
Finish 6 epoch, Loss: 0.491453, Acc: 0.841300
Test Loss: 0.572936, Acc: 0.812800
6250it [05:27, 19.09it/s]
Finish 7 epoch, Loss: 0.412091, Acc: 0.866320
Test Loss: 0.555616, Acc: 0.823200
6250it [05:27, 19.08it/s]
Finish 8 epoch, Loss: 0.353672, Acc: 0.885040
Test Loss: 0.509431, Acc: 0.832100
6250it [05:27, 19.09it/s]
Finish 9 epoch, Loss: 0.289030, Acc: 0.906640
Test Loss: 0.522950, Acc: 0.839600
6250it [05:27, 19.08it/s]
Finish 10 epoch, Loss: 0.250460, Acc: 0.918500
Test Loss: 0.561384, Acc: 0.829100
6250it [05:27, 19.07it/s]
Finish 11 epoch, Loss: 0.208173, Acc: 0.931400
Test Loss: 0.551218, Acc: 0.834500
6250it [05:28, 19.04it/s]
Finish 12 epoch, Loss: 0.177172, Acc: 0.942620
Test Loss: 0.561959, Acc: 0.839100
6250it [05:28, 19.05it/s]
Finish 13 epoch, Loss: 0.144397, Acc: 0.952920
Test Loss: 0.589114, Acc: 0.835700
6250it [05:28, 19.05it/s]
Finish 14 epoch, Loss: 0.123380, Acc: 0.959680
Test Loss: 0.612906, Acc: 0.838100
6250it [05:28, 19.04it/s]
Finish 15 epoch, Loss: 0.105805, Acc: 0.966040
Test Loss: 0.538893, Acc: 0.852600
6250it [05:28, 19.05it/s]
Finish 16 epoch, Loss: 0.083472, Acc: 0.973500
Test Loss: 0.649828, Acc: 0.844600
6250it [05:28, 19.05it/s]
Finish 17 epoch, Loss: 0.074996, Acc: 0.975720
Test Loss: 0.625897, Acc: 0.851300
6250it [05:27, 19.06it/s]
Finish 18 epoch, Loss: 0.066637, Acc: 0.979120
Test Loss: 0.586682, Acc: 0.849300
6250it [05:28, 19.05it/s]
Finish 19 epoch, Loss: 0.058910, Acc: 0.980740
Test Loss: 0.605223, Acc: 0.850100
6250it [05:27, 19.06it/s]
Finish 20 epoch, Loss: 0.047254, Acc: 0.984360
Test Loss: 0.635608, Acc: 0.855900
6250it [05:27, 19.06it/s]
Finish 21 epoch, Loss: 0.045837, Acc: 0.985380
Test Loss: 0.598386, Acc: 0.857500
6250it [05:27, 19.06it/s]
Finish 22 epoch, Loss: 0.040588, Acc: 0.987020
Test Loss: 0.674614, Acc: 0.853300
6250it [05:27, 19.06it/s]
Finish 23 epoch, Loss: 0.032498, Acc: 0.989720
Test Loss: 0.701307, Acc: 0.858800
6250it [05:27, 19.07it/s]
Finish 24 epoch, Loss: 0.028506, Acc: 0.990600
Test Loss: 0.737963, Acc: 0.850900
6250it [05:28, 19.04it/s]
Finish 25 epoch, Loss: 0.026834, Acc: 0.991660
Test Loss: 0.744639, Acc: 0.844400
6250it [05:27, 19.06it/s]
Finish 26 epoch, Loss: 0.030040, Acc: 0.990460
Test Loss: 0.675674, Acc: 0.851300
6250it [05:27, 19.06it/s]
Finish 27 epoch, Loss: 0.026280, Acc: 0.991620
Test Loss: 0.657499, Acc: 0.858600
6250it [05:27, 19.06it/s]
Finish 28 epoch, Loss: 0.023846, Acc: 0.992120
Test Loss: 0.693888, Acc: 0.856000
6250it [05:28, 19.05it/s]
Finish 29 epoch, Loss: 0.020378, Acc: 0.993880
Test Loss: 0.718595, Acc: 0.850700
6250it [05:28, 19.05it/s]
Finish 30 epoch, Loss: 0.024312, Acc: 0.992240
Test Loss: 0.698245, Acc: 0.854200
6250it [05:28, 19.05it/s]
Finish 31 epoch, Loss: 0.021015, Acc: 0.993340
Test Loss: 0.690735, Acc: 0.861600
6250it [05:27, 19.06it/s]
Finish 32 epoch, Loss: 0.017581, Acc: 0.994520
Test Loss: 0.760389, Acc: 0.852300
6250it [05:27, 19.06it/s]
Finish 33 epoch, Loss: 0.018003, Acc: 0.994360
Test Loss: 0.735785, Acc: 0.853100
```

# 迁移学习

```
************************* epoch 1 *************************
1563it [02:42,  9.60it/s]
Finish 1 epoch, Loss: 1.054016, Acc: 0.631980
Test Loss: 0.926332, Acc: 0.675100

************************* epoch 2 *************************
1563it [02:43,  9.58it/s]
Finish 2 epoch, Loss: 0.906125, Acc: 0.682040
Test Loss: 0.891646, Acc: 0.690500

************************* epoch 3 *************************
1563it [02:42,  9.62it/s]
Finish 3 epoch, Loss: 0.836706, Acc: 0.704080
Test Loss: 0.871394, Acc: 0.693400

************************* epoch 4 *************************
1563it [02:43,  9.58it/s]
Finish 4 epoch, Loss: 0.779822, Acc: 0.726940
Test Loss: 0.848277, Acc: 0.704600
```



```python
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets,models
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from tqdm import tqdm   #tqdm是python可扩展的进度条

# 定义超参数start
batch_size = 32 # 批的大小
learning_rate = 0.00001  # 学习率
num_epoches = 20  # 遍历训练集的次数
num_workers =4
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# 定义超参数end

#transform转换
transform = transforms.Compose([
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
vgg =models.vgg16(pretrained=True)
for parma in vgg.parameters():
    parma.requires_grad=False
vgg.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.Dropout(0.5),
    nn.Linear(4096, 10),
)
print(vgg)

model = vgg.to(DEVICE)  #部署到设备上
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=learning_rate)
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

```

