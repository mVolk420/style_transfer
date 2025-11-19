import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_image(image_name, image_size, shape=None):
    if shape is not None:
        resize = transforms.Resize(shape)
    else:
        resize = transforms.Resize(image_size)

    transform = transforms.Compose([
        resize,  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return image.to(device, torch.float)

def show_image(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    tensor_to_image = transforms.ToPILImage()
    image = tensor_to_image(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def save_image(tensor, filename):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    tensor_to_image = transforms.ToPILImage()
    image = tensor_to_image(image)
    image.save(filename)
