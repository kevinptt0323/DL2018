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
import argparse
import matplotlib.pyplot as plt
import time

from models import CVAE, OneHot
from summary import Summary

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                    help='number of epochs')
    parser.add_argument('--model_path', type=str, default=None,
                    help='filename to model')
    parser.add_argument('--history_path', type=str, default=None,
                    help='filename to history')
    parser.add_argument('--eval_epoch', type=int, default=0,
                    help='evaluate every <EVAL_EPOCH> epochs')

    return parser

def parse_opt(args=None):
    parser = get_parser()
    opt = parser.parse_args(args)
    return opt

opt = parse_opt()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

data_shape = dataset[0][0].shape[1:]
data_size = dataset[0][0].numel()

net = CVAE(1, data_shape, data_size, 20, 400, 10)

net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
summary = Summary()
iteration = 0

def loss_fn(inputs, outputs, mean, logvar):
    MSE = F.mse_loss(outputs.view(-1, data_size), inputs.view(-1, data_size), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return MSE + KLD

def train():
    global iteration
    net.train()
    progress = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, mean, logvar = net(inputs, targets)

        loss = loss_fn(inputs, outputs, mean, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item() / inputs.shape[0]

        progress.set_description('Loss: %.6f' % loss_val)

        iteration += 1
        if iteration % 1000 == 0 or iteration == 1:
            i = 0 if iteration == 1 else iteration
            summary.add(i, 'loss', loss_val)

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
epochs = trange(opt.epoch, desc='Epoch', ascii=True)
for epoch in epochs:
    train_loss = train()
    epochs.set_description('Loss: %.6f' % train_loss)
    if opt.eval_epoch > 0 and (epoch + 1) % opt.eval_epoch == 0:
        test()

if opt.history_path:
    summary.write(opt.history_path)

if opt.model_path:
    model_path = os.path.join(opt.model_path)
    torch.save(net.state_dict(), model_path)
