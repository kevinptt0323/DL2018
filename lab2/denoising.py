import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm
from utils import image_to_tensor, tensor_to_image

from models import SkipHourglass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Denoising')
    parser.add_argument('--blind', action='store_true', help='blind')
    parser.add_argument('--steps', default=2400, type=int, help='number of steps')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--input', default='images/noise_image.png', type=str, help='number of iterationa')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    img = image_to_tensor(args.input)
    input_depth = 32
    sigma = 1./30
    noise = torch.rand([1, input_depth] + list(img.shape[2:])) * 0.1 # U(0, 0.1)
    if use_cuda:
        img = img.cuda()
        noise = noise.cuda()

    if args.blind:
        net = SkipHourglass(input_depth, 3, 
                            #down_channels=[8, 16, 32, 64, 128],
                            #up_channels=[8, 16, 32, 64, 128],
                            #skip_channels=[0, 0, 0, 4, 4],
                            down_channels=[128, 128, 128, 128, 128],
                            up_channels=[128, 128, 128, 128, 128],
                            skip_channels=[4, 4, 4, 4, 4],
                            bias=True,
                            upsample_mode='bilinear')
    else:
        pass

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    MSE = torch.nn.MSELoss()

    net_input = Variable(noise)
    target = Variable(img)

    progress = tqdm(range(args.steps))
    for step in progress:
        optimizer.zero_grad()
        net_output = net(net_input)

        loss = MSE(net_output, target)
        loss.backward()
        optimizer.step()

        train_loss = loss.data[0]

        progress.set_description('Loss: %.6f' % train_loss)

        if step % 100 == 0:
            if use_cuda:
                net_output = net_output.cpu()
            tensor_to_image(net_output.data, 'output2/%04d.png' % step)

        if use_cuda:
            noise_new = torch.cuda.FloatTensor(noise.shape).normal_(std=sigma)
        else:
            noise_new = torch.randn(noise.shape) * sigma
        net_input.data += noise_new
