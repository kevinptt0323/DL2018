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
    data = np.transpose(np.array(tensor[0]), (1, 2, 0)).reshape(-1, 3)
    np.random.shuffle(data)
    data = np.transpose(data.reshape([1] + list(tensor.shape[2:]) + [3]), (0, 3, 1, 2))
    return torch.from_numpy(data)

def white_noise(tensor):
    return torch.FloatTensor(tensor.shape).uniform_()
