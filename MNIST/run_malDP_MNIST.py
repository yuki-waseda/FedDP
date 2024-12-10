import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random
import warnings
import csv
import os

warnings.filterwarnings('ignore')

# モデルの定義
class t_model(nn.Module):
    def __init__(self):
        super(t_model, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# クライアントごとのデータサンプリング
def get_samples(num_clients):
    tam = len(mnist_trainset)
    split = int(tam / num_clients)
    indices = list(range(tam))
    samples = []
    for i in range(num_clients):
        t_idx = indices[i * split:(i + 1) * split]
        t_sampler = SubsetRandomSampler(t_idx)
        samples.append(t_sampler)
    return samples

# クライアントクラスの定義
class client():
    def __init__(self, number, loader, state_dict, batch_size=32, epochs=10, lr=0.01):
        self.number = number
        self.model = t_model()
        self.model.load_state_dict(state_dict)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataLoader = loader

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        for e in range(self.epochs):
            self.model.train()
            for images, labels in self.dataLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

# サーバークラス
class server():
    def __init__(self, number_clients):
        self.model = t_model()
        self.n_clients = number_clients
        self.samples = get_samples(self.n_clients)
        self.clients = [client(i, torch.utils.data.DataLoader(mnist_trainset, batch_size=32, sampler=self.samples[i]), self.model.state_dict()) for i in range(number_clients)]
        self.testLoader = torch.utils.data.DataLoader(mnist_testset, batch_size=32)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def eval_acc(self):
        self.model.to(self.device)
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.testLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                _, preds = torch.max(output, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print('Accuracy:', accuracy)
        return accuracy

    def aggregate(self, client_models):
        new_state_dict = {}
        for key in client_models[0].keys():
            new_state_dict[key] = sum(client_model[key] for client_model in client_models) / len(client_models)
        return new_state_dict

    def server_exec(self, mt, max_rounds):
        accuracies = []
        for round_num in range(1, max_rounds + 1):
            print(f'Communication round: {round_num}')
            acc = self.eval_acc()
            accuracies.append(acc)
            selected_clients = random.sample(self.clients, k=mt)
            client_models = [client.update(self.model.state_dict()) for client in selected_clients]
            new_state_dict = self.aggregate(client_models)
            self.model.load_state_dict(new_state_dict)
        return self.model, accuracies

# データセット読み込み
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
num_clients = 30

# 実行結果を保存するファイル名
output_file = "nodp_result.csv"

# CSVファイルが存在しない場合にヘッダーを追加
if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "epsilon", "gamma", "round", "accuracy"])

# 実験パラメータ
epsilon_values = [1, 4, 8]
gamma_values = [0, 0.01, 0.02, 0.03]
num_runs = 3

# 実験の実行
for epsilon in epsilon_values:
    for gamma in gamma_values:
        for run in range(1, num_runs + 1):
            print(f"Run {run}/{num_runs} for epsilon={epsilon}, gamma={gamma}")

            # サーバーインスタンス作成
            serv = server(num_clients)

            # サーバー実行（100ラウンド）
            model, accuracies = serv.server_exec(30, 100)

            # 結果をCSVに保存
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[run, epsilon, gamma, round_num, acc] for round_num, acc in enumerate(accuracies, start=1)])

print(f"\nAll results saved to {output_file}")