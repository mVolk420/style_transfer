import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def load_image(image_name: str, image_size: int, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Loads an image, resizes it, and converts it to a PyTorch tensor.

    Args:
        image_name (str): Path to the image file.
        image_size (int): Size to resize the image to (maintaining aspect ratio) if shape is None.
        shape (Optional[Tuple[int, int]]): Exact (height, width) to resize to. Defaults to None.

    Returns:
        torch.Tensor: The processed image tensor with shape (1, C, H, W).
    """
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

def show_image(tensor: torch.Tensor, title: Optional[str] = None) -> None:
    """
    Displays a PyTorch tensor as an image using matplotlib.

    Args:
        tensor (torch.Tensor): The image tensor to display.
        title (Optional[str]): Title for the plot. Defaults to None.
    """
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    tensor_to_image = transforms.ToPILImage()
    image = tensor_to_image(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def save_image(tensor: torch.Tensor, filename: str) -> None:
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save.
        filename (str): Path to save the image file.
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    tensor_to_image = transforms.ToPILImage()
    image = tensor_to_image(image)
    image.save(filename)
