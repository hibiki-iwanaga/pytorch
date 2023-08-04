# 必要なモジュールを読み込む
# Flask関連
from flask import Flask, render_template, request, redirect, url_for, abort
from PIL import Image, ImageOps
from datetime import datetime
# PyTorch関連
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.functional as f
import torchvision
from torchvision import datasets, transforms


transform = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Resize((26,26)),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class RegressionNet(torch.nn.Module):

    def __init__(self):
        super(RegressionNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 *5 * 5, 1024)
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
    


device = torch.device("cpu")

# 学習モデルをロードする

model = RegressionNet()
model.load_state_dict(torch.load('./model.pt'))

# model = model.eval()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)
        # 画像ファイルを読み込む
        image = Image.open(filepath)
        # PyTorchで扱えるように変換(リサイズ、白黒反転、正規化、次元追加)
        image = transform(image)
        
        outputs = model(image)
        # 予測を実施
        result = int(outputs.tolist()[0][0])

        return render_template("index.html", filepath=filepath, result=result)


if __name__ == "__main__":
    app.run(debug=True)
