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

from models import Generator, Discriminator, OneHot
from summary import Summary

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
    parser.add_argument('--lr_g', type=float, default=1e-3,
                    help='learning rate of generator')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                    help='learning rate of discriminator')
    parser.add_argument('--epoch', type=int, default=100,
                    help='number of epochs')
    parser.add_argument('--name', type=str, default=None,
                    help='name to current training process')
    parser.add_argument('--eval_epoch', type=int, default=0,
                    help='evaluate every <EVAL_EPOCH> epochs')
    parser.add_argument('--z_size', type=int, default=64, help='latent z size')
    parser.add_argument('--g_channel', type=int, default=64, help='g_channel')
    parser.add_argument('--d_channel', type=int, default=64, help='d_channel')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')

    return parser

def parse_opt(args=None):
    parser = get_parser()
    opt = parser.parse_args(args)
    assert opt.name is None or opt.eval_epoch > 0, "Should use --name flag along with --eval_epoch"
    return opt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)

opt = parse_opt()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor()])

dataset = torchvision.datasets.MNIST(root='../data', transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

in_channels = 1
c_size = 10

netG = Generator(opt.z_size, opt.g_channel, in_channels).to(device)
netD = Discriminator(c_size, opt.d_channel, in_channels).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

crit_bce = nn.BCELoss().to(device)
crit_ce = nn.CrossEntropyLoss().to(device)

fixed_noise = torch.randn(opt.batch_size, opt.z_size, 1, 1, device=device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(list(netG.parameters()) + list(netD.Q.parameters()), lr=opt.lr_g, betas=(opt.beta1, 0.999))

iteration = 0
summary = Summary()
output_dir = None
if opt.name:
    output_dir = os.path.join('data', opt.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def train():
    global iteration
    netG.train()
    netD.train()
    progress = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (inputs, _) in progress:
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        inputs = inputs.to(device)
        optimizerD.zero_grad()
        batch_size = inputs.shape[0]
        label = torch.full((batch_size,), real_label, device=device)

        output, output_q = netD(inputs)
        lossD_real = crit_bce(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        # train with fake
        label_c = torch.randint(0, c_size, (batch_size,), dtype=torch.long, device=device)
        label_onehot = OneHot(label_c, c_size)
        noise = torch.cat([torch.randn(batch_size, opt.z_size - c_size, device=device),
                           label_onehot], dim=1)
        noise = noise.unsqueeze(2).unsqueeze(3)
        fake = netG(noise)
        label.fill_(fake_label)

        output, output_q = netD(fake.detach())
        lossD_fake = crit_bce(output, label)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        optimizerG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output, output_q = netD(fake)
        lossG_reconstruct = crit_bce(output, label)
        lossG_mi = crit_ce(output_q, label_c)
        lossG = lossG_reconstruct + lossG_mi
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        progress.set_description('lossD: {:6f} ({:6f} {:6f}) lossG: {:6f} ({:6f} {:6f})' \
                                 .format(lossD.item(), lossD_real.item(), lossD_fake.item(),
                                         lossG.item(), lossG_reconstruct.item(), lossG_mi.item()))

        iteration += 1
        if iteration % 100 == 0 or iteration == 1:
            i = 0 if iteration == 1 else iteration
            summary.add(i, 'lossD', lossD.item())
            summary.add(i, 'lossG', lossG.item())
        # if iteration % 100 == 0 or iteration == 1:
        #     print(F.softmax(output_q[0], dim=0))
        #     print(label_c[0])

def test():
    netG.eval()
    netD.eval()
    with torch.no_grad():
        plt.figure(figsize=(5,10))
        noise = torch.rand(c_size, opt.z_size - c_size, device=device)
        noise = torch.cat([noise] * c_size, dim=1).view(-1, opt.z_size - c_size)
        label = torch.arange(c_size, dtype=torch.long, device=device)
        label = torch.cat([label] * c_size)
        onehot = OneHot(label, c_size)
        input = torch.cat([noise, onehot], dim=1).unsqueeze(2).unsqueeze(3)

        outputs = netG(input)
        imgs = outputs.view(-1, 64, 64).cpu()

        fig, ax = plt.subplots(nrows=c_size, ncols=c_size, figsize=(10, 10), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        for idx, img in enumerate(torch.unbind(imgs)):
            axx = ax[idx // c_size][idx % c_size]
            axx.axis('off')
            axx.imshow(img, cmap='gray')

        fig_dir = os.path.join(output_dir, 'fig')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, '%i.png' % epoch))
        plt.clf()
        plt.close()

epochs = trange(opt.epoch, desc='Epoch', ascii=True)
for epoch in epochs:
    train()
    if output_dir and opt.eval_epoch > 0 and (epoch + 1) % opt.eval_epoch == 0:
        test()

if output_dir:
    summary.write(os.path.join(output_dir, 'result.csv'))
    netG_path = os.path.join(output_dir, 'netG.pth')
    netD_path = os.path.join(output_dir, 'netD.pth')
    torch.save(netG.state_dict(), netG_path)
    torch.save(netD.state_dict(), netD_path)
