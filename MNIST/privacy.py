import torchvision
import os
import csv
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
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

p_budget = 0.001
mt = 30
clients = 100


for epsilon in [1,2,4,8]:
    
    sigmat = np.sqrt(2 * np.log(1.25 / p_budget)) * 1 /epsilon +1.12 
    print('eps: ',epsilon)
    print('eps: ',sigmat)
    i=1
    while(1):
        if (i>100):
            break
        orders = ( [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                    list(range(5, 64)) + [128, 256, 512])
        rdp = compute_rdp(mt/clients, sigmat/2, i, orders)
        _,delta_spent, opt_order = get_privacy_spent(orders, rdp, target_eps=epsilon)
        
        print('Round: ',i)
        print('Delta spent: ', delta_spent)
        print('Delta budget: ', p_budget)  
        i+=1        


