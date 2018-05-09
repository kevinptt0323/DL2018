import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange
import os

from models import CVAE, OneHot
from summary import Summary

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

data_shape = (28, 28)
data_size = 28 * 28

net = CVAE(1, data_shape, data_size, 20, 400, 10)
net = net.to(device)

state_dict = torch.load('data/model2.pth')
net.load_state_dict(state_dict)

summary = Summary()
iteration = 0

def test():
    net.eval()
    noise = torch.rand(10, 20).to(device)
    label = torch.arange(0, 10).to(device)
    onehot = OneHot(label, 10)
    outputs = net.decoder(noise, onehot)
    
for epoch in trange(100, desc='Epoch', ascii=True):
    test()

