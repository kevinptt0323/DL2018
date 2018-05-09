import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import os

from models.CVAE import CVAE
from summary import Summary

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

data_shape = dataset[0][0].shape[1:]
data_size = dataset[0][0].numel()

net = CVAE(1, data_shape, data_size, 400, 20, 10)

net = net.to(device)

if use_cuda:
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    pass

optimizer = optim.Adam(net.parameters(), lr=1e-3)
summary = Summary()
iteration = 0

def train():
    global iteration
    net.train()
    progress = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, *_ = net(inputs, targets)
        loss = F.mse_loss(outputs, inputs.view(-1, data_size), size_average=False)
        loss.backward()
        optimizer.step()

        progress.set_description('Loss: %.6f' % loss.item())

        iteration += 1
        if iteration % 50 == 0:
            summary.add(iteration, 'loss', loss.item())
    
for epoch in trange(100, desc='Epoch', ascii=True):
    train()

summary.write("csv/history1.csv")

model_path = os.path.join('data/model1.pth')
torch.save(net.to(torch.device('cpu')).state_dict(), model_path)
