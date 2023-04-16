import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import csv
import torch
import torchvision


#データセットクラスの定義
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, label_path, transform=None):
        x = []
        y = []
        file = open(label_path, 'r')
        data = csv.reader(file)
        for row in data:
            x.append(row[0])
            y.append(float(row[1]))
        file.close()
    
        self.x = x    
        self.y = torch.from_numpy(np.array(y)).float().view(-1, 1)
     
        self.transform = transform
  
  
    def __len__(self):
        return len(self.x)
  
  
    def __getitem__(self, i):
        img = PIL.Image.open(self.x[i]).convert('RGB')
        if self.transform is not None:
              img = self.transform(img)
    
        return img, self.y[i]
    
    
    
transform = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Resize(26),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#データセット読み込み
train_data_dir = './csv/effspin5_train.csv'
valid_data_dir = './csv/effspin5_val.csv'
test_data_dir = './csv/effspin5_test.csv'
data_dir = './csv/actual_test.csv'''
trainset = MyDataset(train_data_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
validset = MyDataset(valid_data_dir, transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=False)
testset = MyDataset(test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
gwset = MyDataset(data_dir, transform=transform)
gwloader = torch.utils.data.DataLoader(gwset, batch_size=1, shuffle=False)




#回帰モデル定義
class RegressionNet(torch.nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 5 * 5, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1)


    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 32 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

net = RegressionNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)


#最適化関数と損失関数の
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


import time
num_epocs = 50

list1=[]
list2=[]

train_loss = []
valid_loss = []
time_sta = time.time()
for epoch in range(num_epocs):
  # 学習
    net.train()
    running_train_loss = 0.0
    running_acc = 0.0
    with torch.set_grad_enabled(True):
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            a = outputs.tolist()
            b = labels.tolist()
            list1.append(a)
            list2.append(b)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()

    train_loss.append(running_train_loss / len(trainloader))
  
  # 検証
    net.eval()
    running_valid_loss = 0.0
    val_running_acc = 0.0
    with torch.set_grad_enabled(False):
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item()

    valid_loss.append(running_valid_loss / len(validloader))

    print('#epoch:{}\ttrain loss: {}\tvalid loss: {}'.format(epoch,
                                                running_train_loss / len(trainloader), 
                                                running_valid_loss / len(validloader)))
    
    
    
    
test_loss=[]
running_test_loss = 0.0
pred=[]
ans=[]
with torch.set_grad_enabled(False):
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        list1 = labels.tolist()
        outputs = net(inputs)
        list2 = outputs.tolist()
        for i in range(len(list1)):
            ans.append(list1[i])
        for i in range(len(list2)):
            pred.append(list2[i])
        loss = criterion(outputs, labels)
        running_test_loss += loss.item()

test_loss.append(running_test_loss / len(testloader))

print('test loss: {}'.format(running_test_loss / len(testloader)))
                
    
import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
            
            
ans= list(flatten(ans))
pred= list(flatten(pred))

error=[]
for i in range(len(ans)):
    error.append((pred[i]/ans[i])-1)
    
    
    
import statistics
print(max(error))
print(min(error))
print(statistics.mean(error))
print(statistics.stdev(error))


error2=[]
for i in range(len(error)):
    if -1<=error[i]<=1:
        error2.append(error[i])
print(len(error))
print(len(error2))


print(max(error2))
print(min(error2))
print(statistics.mean(error2))
print(statistics.stdev(error2))

plt.hist(error2,bins=60)
plt.xlabel('誤差（%）',fontsize=16)
plt.ylabel('度数',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.hist(error,bins=100,range=(-5,5))
plt.xlabel('誤差（%）',fontsize=16)
plt.ylabel('度数',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()