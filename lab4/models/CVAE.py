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
    def __init__(self, in_channel, data_shape, data_size, enc_size, hid_size, label_len, bias=True):
        super(Encoder, self).__init__()

        self.data_shape = data_shape
        self.data_size = data_size

        self.conv11 = nn.Conv2d(in_channel + label_len, 1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.fc1 = nn.Linear(data_size, hid_size, bias=bias)
        self.fc21 = nn.Linear(hid_size, enc_size, bias=bias)
        self.fc22 = nn.Linear(hid_size, enc_size, bias=bias)

    def forward(self, x, onehot):
        onehot = torch.stack([onehot] * self.data_size, 2).view(*onehot.shape, *self.data_shape)
        x = torch.cat([x, onehot], dim=1)
        h1 = self.conv11(x).view(-1, self.data_size)
        h1 = F.relu(self.fc1(h1))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Module):
    def __init__(self, data_size, enc_size, hid_size, label_len, bias=True):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(enc_size + label_len, hid_size, bias=bias)
        self.fc4 = nn.Linear(hid_size, data_size, bias=bias)
        # self.conv21 = nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, z, onehot):
        x = torch.cat([z, onehot], dim=1)
        h3 = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(h3))

class CVAE(nn.Module):
    def __init__(self, in_channel, data_shape, data_size, enc_size, hid_size, label_len, bias=True):
        super(CVAE, self).__init__()

        self.data_shape = data_shape
        self.data_size = data_size
        self.label_len = label_len

        self.encoder = Encoder(in_channel, data_shape, data_size, enc_size, hid_size, label_len, bias)
        self.decoder = Decoder(data_size, enc_size, hid_size, label_len, bias)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, x, label):
        onehot = OneHot(label, self.label_len)
        mu, logvar = self.encoder(x, onehot)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, onehot), mu, logvar
