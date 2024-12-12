# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:08:45 2021

@author: tamji
"""

import torchvision
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import csv
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from IPython.display import clear_output
from numpy import linalg as LA
from rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from rdp_accountant import get_privacy_spent
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import random

#%%

#モデルの定義
class t_model(nn.Module):
    def __init__(self):
        super(t_model, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)
    
#クライアントごとのデータサンプリング
#Return the samples that each client is going to have as a private training data set. This is a not overlapping set
def get_samples(num_clients):
    tam = len(mnist_trainset)
    split= int(tam/num_clients)
    split_ini = split
    indices = list(range(tam))
    init=0
    samples = []
    for i in range(num_clients):     
        t_idx = indices[init:split]
        t_sampler = SubsetRandomSampler(t_idx)
        samples.append(t_sampler)
        init = split
        split = split+split_ini
    return samples 
      

#%%

#クライアントクラスの定義
class client():
    def __init__(self, number, loader, state_dict, batch_size = 32, epochs=10, lr=0.01):
        self.number = number
        self.model = t_model()
        self.model.load_state_dict(state_dict)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        #修正
        #self.device =  device =  torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataLoader = loader                                       
                                           
                                  
  #
    def update(self, state_dict):
        w0 = state_dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        running_loss = 0
        accuracy = 0
        for e in range(self.epochs):
            # Model in training mode, dropout is on
            self.model.train()
            accuracy=0
            running_loss = 0
            for images, labels in self.dataLoader:            
                images, labels = images.to(self.device), labels.to(self.device)                       
                self.optimizer.zero_grad()            
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()            
                running_loss += loss.item()        
        S ={} 
        wt1 = {}
        for key, value in w0.items():
            wt1[key] = self.model.state_dict()[key]  - value   
            S[key] = LA.norm(wt1[key].cpu(), 1)
        return wt1, S
#%%

#画像とクラス確率の出力
def view_classify(img, ps):
    ## 修正
    ##ps = ps.data.numpy().squeeze()    
    ps = ps.cpu().data.numpy().squeeze() 
    img = img.cpu()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    
#後述
def uniform_proposal(x, delta=2.0):
    return np.random.uniform(x - delta, x + delta)

#後述
def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 1 # start somewhere

    for i in range(nsamples):
        trial = proposal(x) # random neighbour from the proposal distribution
        acceptance = p(trial)/p(x)

        # accept the move conditionally
        if np.random.uniform() < acceptance:
            x = trial
        yield x

#ガウスノイズ生成
def noiseGen(mu, sigma, suma):
    mu = mu
    sigma = sigma
    p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)
    if len(suma.shape)==2:
        u,v = suma.shape
    else:
        u = suma.shape
        v = 1
    samples = np.array(metropolis_sampler(p, u*v))
#     ranSample = random.sample(samples,u*v)
#     ranSampleArr = np.array(ranSample)
    print(samples.shape)
    print(len(samples))
    print(samples)
    noise = samples.reshape((u,v))
    return noise
#%%

#@title
#サーバークラス
class server():
    def __init__(self, number_clients, p_budget, epsilon,gamma):
        #sigmat = 1.12
        self.model = t_model()
        #sigmat = 0.55 * np.sqrt(2 * np.log(1.25 / p_budget)) * 1 / epsilon
        sigmat = np.sqrt(2 * np.log(1.25 / p_budget)) * 1 / epsilon +1.12
        #sigmat =  np.sqrt(2 * np.log(1.25 / p_budget)) * 1 / epsilon
        self.sigmat = sigmat   
        self.n_clients = number_clients
        self.gamma = gamma
        self.samples = get_samples(self.n_clients)
        self.clients = list()
        for i in range(number_clients):
            loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, sampler=self.samples[i])
            self.clients.append(client(i, loader, self.model.state_dict()))
        self.p_budget = p_budget
        self.epsilon = epsilon
        self.testLoader = torch.utils.data.DataLoader(mnist_testset, batch_size=32)
        #修正
        #self.device = torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])

    #テストデータに対する精度
    #Evaluates the accuracy of the current model with the test data.  
    def eval_acc(self):
        self.model.to(self.device)
        #print('Aqui voy!')
        running_loss = 0
        accuracy = 0
        self.model.eval()
        suma=0
        total = 0
        running_loss = 0
        for images, labels in self.testLoader:            
            images, labels = images.to(self.device), labels.to(self.device) 
            output = self.model.forward(images)             
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total += equals.size(0)
            suma = suma + equals.sum().item()
        else:
            accuracy = suma/float(total)
            print('Accuracy: ',suma/float(total))
        return accuracy



    #Given:
    # mt: number of clients involved in this round. 
    #deltas: list of dicts with the deltas of every client
    #norms: list of dicts with the norms of every weights for every client
    #sigma: Sigma to calculate the StdDistribution of the GaussianNormalNoise
    #state_dicst: Dict with the current model weights 

    # This functions apply noise to the given deltas. 
    #差分プライバシー適用
    def sanitaze(self,mt, deltas, norms, sigma, state_dict, gamma):    
        new_dict = {}
        inclMalSum_dict = {}
        malModel = [5]
        prom = 1/float(mt)   

        sanitized_deltas = [{} for _ in range(len(deltas))] 
        for key, value in state_dict.items():
            #print(len(deltas))
            #print(deltas[0][key])
            S=[]
            for i in range(len(norms)):        
                S.append(norms[i][key])
            S_value = np.median(S)      
            wt = value

            for i in range(len(deltas)):    
                clip = (max(1, float(norms[i][key]/S_value)))   
                clippedDelta = ((deltas[i][key])/clip)
                if any(i < m for m in malModel) :
                    noise = (np.random.normal((np.sqrt(2*gamma)*(sigma*S_value)/30), float((sigma**2)*(S_value**2)/30), size = deltas[i][key].shape))
                
                else: 
                    noise = (np.random.normal(0, float((sigma**2)*(S_value**2)/30), size = deltas[i][key].shape))
                clippedDelta = clippedDelta.cpu().numpy()
                modelSum = clippedDelta + noise
                sanitized_deltas[i][key] = torch.from_numpy(modelSum).float().to('cpu')
            
#             noise = np.random.normal(0, float(S_value * sigma), size = suma.shape)
            
            #if (len(suma.shape)==2):
            #    u,v = suma.shape
            #    mu = 0
            #    sigma = float(S_value * sigma)
            #    p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu - np.sqrt(2*gamma)*sigma)**2)/2./sigma/sigma)
            #    #p2 = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)
            #    samples1 = list(metropolis_sampler(p, u*v))
            #    sample1 = np.array(samples1)
            #    noise1 = sample1.reshape((u,v))
            #else:
            #    u = len(suma)
            #    mu = 0
            #    sigma = float(S_value * sigma)
            #    p = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/2./sigma/sigma)
            #    #p2 = lambda x: 1./sigma/np.sqrt(2*np.pi)*np.exp(-((x-mu - np.sqrt(2*gamma)*sigma)**2)/2./sigma/sigma)
            #    samples2 = list(metropolis_sampler(p, u))
            #    noise1 = np.array(samplesxx)
                
            #noise = noise1
        for key, value in state_dict.items():
            wt = value
            inclMalSum = 0
            for i in range(mt):
                inclMalSum = inclMalSum + sanitized_deltas[i][key] 
            inclMalSum = inclMalSum*prom
            inclMalSum = wt + inclMalSum
            inclMalSum_dict[key] = inclMalSum
        return inclMalSum_dict
        

    #学習ラウンドの繰り返し
    def server_exec(self,mt):    
        i=1
        testLossList = []
        while(1):
#             clear_output()
            print('Comunication round: ', i)
            test_loss = self.eval_acc()         
            testLossList.append(test_loss) 
            rdp = compute_rdp(float(mt/len(self.clients)), self.sigmat, i, self.orders)
            _,delta_spent, opt_order = get_privacy_spent(self.orders, rdp, target_eps=self.epsilon)
            print('Delta spent: ', delta_spent)
            print('Delta budget: ', self.p_budget)  
            
            #if self.epsilon== 1:
            #    if self.p_budget < delta_spent:
            #        break
            if 50<i:
                break
            Zt = np.random.choice(self.clients, mt)      
            deltas = []
            norms = []
            for client in Zt:
                deltaW, normW = client.update(self.model.state_dict())   
                deltas.append(deltaW)
                norms.append(normW)     
            self.model.to('cpu')
            new_state_dict = self.sanitaze(mt, deltas, norms, self.sigmat, self.model.state_dict(),self.gamma)
            self.model.load_state_dict(new_state_dict)
            i+=1
        return self.model,testLossList
            
#             images, labels = next(iter(valloader))
#             img = images[0].view(1, 784)
#             # Turn off gradients to speed up this part
#             with torch.no_grad():
#                 logps = self.model(img)

#             # Output of the network are log-probabilities, need to take exponential for probabilities
#             ps = torch.exp(logps)
#             probab = list(ps.numpy()[0])
#             print("Predicted Digit =", probab.index(max(probab)))
#             view_classify(img.view(1, 28, 28), ps)
#%%
#学習開始
#データセット読み込み・サーバーインスタンス作成
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#修正
#device =  torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_clients = 30
train_len = len(mnist_trainset)
test_len = len(mnist_testset)
valloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

# 実行結果を保存するファイル名
output_file = "divmaldp_result.csv"

# CSVファイルが存在しない場合にヘッダーを追加
if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "epsilon", "gamma", "round", "accuracy"])

# 実験パラメータ
epsilon_values = [1, 4, 8]
gamma_values = [0, 0.01, 0.02, 0.03]
num_runs = 3
p_budget = 0.001

# 実験の実行
for epsilon in epsilon_values:
    for gamma in gamma_values:
        for run in range(1, num_runs + 1):
            print(f"Run {run}/{num_runs} for epsilon={epsilon}, gamma={gamma}")

            # サーバーインスタンス作成
            serv = server(num_clients, p_budget,epsilon,gamma)

            # サーバー実行（100ラウンド）
            model, accuracies = serv.server_exec(30)

            # 結果をCSVに保存
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([[run, epsilon, gamma, round_num, acc] for round_num, acc in enumerate(accuracies, start=1)])

print(f"\nAll results saved to {output_file}")