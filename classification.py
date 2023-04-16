import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,))
])

#画像読み込み、確認
from pathlib import Path
train_image_dir = Path().cwd().parent/'MLimages/images6/train'
val_image_dir = Path().cwd().parent/'MLimages/images6/val'
train_dataset = torchvision.datasets.ImageFolder(root=train_image_dir, transform=transform)
validation_dataset = torchvision.datasets.ImageFolder(root=val_image_dir, transform=transform)
image1,label1 = train_dataset[0]
print(image1.shape)
dataset = torchvision.datasets.ImageFolder(root=train_image_dir)
image,label = dataset[1003]
image2,label2 = dataset[0]
print(len(dataset))
plt.figure(figsize=(3,3))
plt.axis('off')
plt.imshow(image)
plt.show()
plt.figure(figsize=(3,3))
plt.axis('off')
plt.imshow(image2)
plt.show()


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


#CNNのモデル定義
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
    
#モデルの学習と検証
num_epocs = 15
losses = []
accs = []
val_losses = []
val_accs = []
for epoch in range(num_epocs):
    # 学習
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        optimizer.step()
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    
    # 検証
    val_running_loss = 0.0
    val_running_acc = 0.0
    for val_imgs, val_labels in validation_dataloader:
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        val_output = model(val_imgs)
        val_loss = criterion(val_output, val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_output, dim=1)
        val_running_acc += torch.mean(val_pred.eq(val_labels).float())
    val_running_loss /= len(validation_dataloader)
    val_running_acc /= len(validation_dataloader)
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)
    print("epoch: {}, loss: {}, acc: {}    " \
    "val_epoch: {}, val_loss: {}, val_acc: {}".format(epoch, running_loss, running_acc, epoch, val_running_loss, val_running_acc))
    
    
    
#結果確認（loss）
plt.plot(losses, label='train loss')
plt.plot(val_losses, label='validation loss')
plt.legend()

# 結果確認（accuracy)
plt.plot(accs, label='train acc')
plt.plot(val_accs, label='validation acc')
plt.legend()
