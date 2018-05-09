import torch
import torch.nn as nn
import torch.nn.functional as F

def OneHot(idx, n):
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.shape[0], n, device=idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot

class Encoder(nn.Module):
    def __init__(self, in_channel, data_shape, data_size, latent_size, hid_size, label_len, bias=True):
        super(Encoder, self).__init__()

        self.data_shape = data_shape
        self.data_size = data_size

        self.conv11 = nn.Conv2d(in_channel + label_len, 1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.fc = nn.Linear(data_size, hid_size, bias=bias)
        self.fc_mean = nn.Linear(hid_size, latent_size, bias=bias)
        self.fc_logvar = nn.Linear(hid_size, latent_size, bias=bias)

    def forward(self, x, onehot):
        onehot = torch.stack([onehot] * self.data_size, 2).view(*onehot.shape, *self.data_shape)
        x = torch.cat([x, onehot], dim=1)
        out = F.relu(self.conv11(x))
        out = out.view(-1, self.data_size)
        out = F.relu(self.fc(out))
        return self.fc_mean(out), self.fc_logvar(out)

class Decoder(nn.Module):
    def __init__(self, data_shape, data_size, latent_size, label_len, bias=True):
        super(Decoder, self).__init__()
        self.data_shape = data_shape
        self.data_size = data_size

        self.fc = nn.Linear(latent_size + label_len, data_size // 2, bias=bias)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(2, label_len+1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(label_len+1, 1, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.conv3 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, z, onehot):
        x = torch.cat([z, onehot], dim=1)
        out = F.relu(self.fc(x))
        out = out.view(-1, 2, self.data_shape[0] // 2, self.data_shape[1] // 2)
        out = F.relu(self.conv1(out))
        out = self.upsample(out)
        out = F.relu(self.conv2(out))
        # out = F.sigmoid(self.conv3(out))
        return out

class CVAE(nn.Module):
    def __init__(self, in_channel, data_shape, data_size, latent_size, hid_size, label_len, bias=True):
        super(CVAE, self).__init__()

        self.data_shape = data_shape
        self.data_size = data_size
        self.label_len = label_len

        self.encoder = Encoder(in_channel, data_shape, data_size, latent_size, hid_size, label_len, bias)
        self.decoder = Decoder(data_shape, data_size, latent_size, label_len, bias)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps * std + mean
        else:
            return mean

    def forward(self, x, label):
        onehot = OneHot(label, self.label_len)
        mean, logvar = self.encoder(x, onehot)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z, onehot), mean, logvar
