import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import os

from models import CVAE
from summary import Summary

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

data_shape = dataset[0][0].shape[1:]
data_size = dataset[0][0].numel()

net = CVAE(1, data_shape, data_size, 20, 400, 10)

net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
summary = Summary()
iteration = 0
min_loss = 1e9

def train():
    global iteration, min_loss
    net.train()
    progress = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mean, logvar = net(inputs, targets)
        MSE = F.mse_loss(outputs.view(-1, data_size), inputs.view(-1, data_size), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = MSE + KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_description('Loss: %.6f' % loss.item())
        min_loss = min(min_loss, loss.item())

        iteration += 1
        if iteration % 50 == 0:
            summary.add(iteration, 'loss', loss.item())

epochs = trange(100, desc='Epoch', ascii=True)
for epoch in epochs:
    train()
    epochs.set_description('Loss: %.6f' % min_loss)

summary.write("csv/history2.csv")

model_path = os.path.join('data/model2.pth')
torch.save(net.state_dict(), model_path)
