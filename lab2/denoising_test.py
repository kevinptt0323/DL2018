import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
from utils import *
from PIL import Image
from skimage.measure import compare_psnr

from models import SkipHourglass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Denoising')
    parser.add_argument('--step', default=2400, type=int, help='number of steps')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--input', default='images/noise_GT.png', type=str, help='path to input')
    parser.add_argument('--output', type=str, help='filename to ouptut loss in csv')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    input_depth = 3
    sigma = 1./30
    origin_image = image_to_tensor(args.input)
    input_size = torch.Size([1, input_depth] + list(origin_image.shape[2:]))

    transforms = [
        ('Image', lambda t: t.clone()),
        ('Image + Noise', lambda t: add_noise(t, std=25/255.)),
        ('Image shuffled', random_shuffle),
        ('U(0 1) noise', white_noise),
    ]

    csv_arr = { }

    for transform_name, transform in transforms:
        target = transform(origin_image)
        noise = torch.FloatTensor(input_size).uniform_(0, 0.1) # U(0, 0.1)
        if use_cuda:
            target = target.cuda()
            noise = noise.cuda()

        net = SkipHourglass(input_depth, 3,
                            down_channels=[8, 16, 32, 64, 128],
                            up_channels=[8, 16, 32, 64, 128],
                            skip_channels=[0, 0, 0, 4, 4],
                            bias=True,
                            upsample_mode='bilinear')

        if use_cuda:
            net.cuda()

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        MSE = nn.MSELoss()

        net_input = Variable(noise)
        target = Variable(target)

        progress = tqdm(range(1, args.step+1))
        psnr = 0
        max_psnr = 0
        csv_arr[transform_name] = []
        for step in progress:
            optimizer.zero_grad()
            net_output = net(net_input)

            loss = MSE(net_output, target)
            loss.backward()
            optimizer.step()

            train_loss = loss.data[0]

            progress.set_description('%-15s | Loss: %.6f' % (transform_name, train_loss))

            csv_arr[transform_name].append(train_loss)

            if use_cuda:
                noise_new = torch.cuda.FloatTensor(noise.shape).normal_(std=sigma)
            else:
                noise_new = torch.FloatTensor(noise.shape).normal_(std=sigma)

            net_input.data += noise_new

    if args.output:
        with open(args.output, 'w') as f:
            f.write(''.join([(',' + name) for name, _ in transforms]))
            f.write('\n')
            for i in range(args.step):
                f.write('%d' % (i+1))
                f.write(''.join([(',%f' % csv_arr[name][i]) for name, _ in transforms]))
                f.write('\n')

