from PIL import Image
import numpy as np
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
    pil.save(filename)
    return pil