import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import os
import matplotlib.pyplot as plt
import time

from models import CVAE, OneHot
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

        loss_val = loss.item() / inputs.shape[0]

        progress.set_description('Loss: %.6f' % loss_val)
        min_loss = min(min_loss, loss_val)

        iteration += 1
        if iteration % 50 == 0:
            summary.add(iteration, 'loss', loss_val)

    return loss_val

def test():
    net.eval()

    plt.figure(figsize=(5,10))
    noise = torch.rand(10, 20).to(device)
    label = torch.arange(10, dtype=torch.long).to(device)
    onehot = OneHot(label, 10)
    outputs = net.decoder(noise, onehot)
    for p in range(10):
        plt.subplot(5,2,p+1)
        plt.text(0,0,"label=%i"%label[p].item(), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(outputs[p].view(28,28).data.cpu().numpy(), cmap='gray')
        plt.axis('off')

    fig_root = './data/fig'

    if not os.path.exists(os.path.join(fig_root, str(ts))):
        os.makedirs(os.path.join(fig_root, str(ts)))

    plt.savefig(os.path.join(fig_root, str(ts), "%i.png"%epoch), dpi=300)
    plt.clf()
    plt.close()

ts = int(time.time())
epochs = trange(100, desc='Epoch', ascii=True)
for epoch in epochs:
    train_loss = train()
    epochs.set_description('Loss: %.6f' % train_loss)
    if (epoch + 1) % 10 == 0:
        test()

summary.write("csv/history1.csv")

model_path = os.path.join('data/model/model1.pth')
torch.save(net.state_dict(), model_path)
