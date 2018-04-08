from PIL import Image
import numpy as np
import torch
import torchvision

def image_to_tensor(filename):
    img = Image.open(filename)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = pil_to_tensor(img)
    return tensor.view([1]+list(tensor.shape))

def tensor_to_image(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    if filename:
        pil.save(filename)
    return pil

def add_noise(tensor, std=1):
    noise_new = torch.FloatTensor(tensor.shape).normal_(std=std)
    return torch.clamp(tensor + noise_new, min=0, max=1)

def random_shuffle(tensor):
    perm1 = torch.randperm(tensor.shape[2])
    perm2 = torch.randperm(tensor.shape[3])
    return tensor[:,:,perm1,:][:,:,:,perm2]

def white_noise(tensor):
    return torch.FloatTensor(tensor.shape).uniform_()
