import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)//2, stride=2,
                               **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)//2, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode, **kwargs):
        super(UpBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)//2, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn0(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)//2, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SkipHourglass(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 down_channels, up_channels, skip_channels,
                 down_kernel=3, up_kernel=3, skip_kernel=1,
                 bias=False,
                 upsample_mode='bilinear'):

        super(SkipHourglass, self).__init__()
        
        self.layer_num = len(down_channels)
        self.upsample_mode = upsample_mode

        if not isinstance(down_kernel, list):
            down_kernel = [down_kernel] * self.layer_num

        if not isinstance(up_kernel, list):
            up_kernel = [up_kernel] * self.layer_num

        if not isinstance(skip_kernel, list):
            skip_kernel = [skip_kernel] * self.layer_num

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for i in range(self.layer_num):
            if i == 0:
                prev_ch = in_channels
            else:
                prev_ch = down_channels[i - 1]

            down = DownBlock(prev_ch, down_channels[i], down_kernel[i],
                             bias=bias)
            self.down_layers.append(down)

            if skip_channels[i] > 0:
                skip = SkipBlock(prev_ch, skip_channels[i], skip_kernel[i],
                                 bias=bias)
            else:
                skip = None
            self.skip_layers.append(skip)

        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                prev_ch = down_channels[i]
            else:
                prev_ch = up_channels[i + 1]

            up = UpBlock(prev_ch + skip_channels[i], up_channels[i],
                         up_kernel[i], mode=upsample_mode, bias=bias)
            self.up_layers.append(up)

        self.out_conv = nn.Conv2d(up_channels[0], out_channels, 1, bias=bias)

    def forward(self, x):
        out = x
        skips = []
        for i in range(self.layer_num):
            if not self.skip_layers[i] is None:
                skip = self.skip_layers[i](out)
            else:
                skip = None
            skips.append(skip)
            out = self.down_layers[i](out)
        
        for i in range(self.layer_num-1, -1, -1):
            out = F.upsample(out, scale_factor=2, mode=self.upsample_mode)
            if not skips[i] is None:
                out = torch.cat([out, skips[i]], 1)
            out = self.up_layers[i](out)

        out = F.sigmoid(self.out_conv(out))
        return out

