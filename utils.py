import os
import functools
from time import time

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import PIL


def func_timer(func):
    """Decorator that measures the wrapped function's execution time."""
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start_time = time()
        out = func(*args, **kwargs)
        duration = time() - start_time
        print(f"<{func.__name__}> run time: {duration:.4f} s")
        return out
    return _measure_time

def save_model(model, checkpoint_path):
    model.eval()
    model_state_dict = model.state_dict()
    torch.save({'model_state_dict' : model_state_dict,
                }, checkpoint_path)

def prepare_val_folder(dataset_path):
    """
    Split validation images into separate class-specific sub folders. Like this the
    validation dataset can be loaded as an `ImageFolder`.
    """
    val_dir = os.path.join(dataset_path, 'val')
    img_dir = os.path.join(val_dir, 'images')

    # read csv file that associates each image with a class
    annotations_file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = annotations_file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    annotations_file.close()

    # create class folder if not present and move image into it
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(os.path.dirname(img_dir), folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

    # remove old image path
    if os.path.exists(img_dir):
        os.rmdir(img_dir)

class BicubicUpsampling():
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img.resize((self.size, self.size), PIL.Image.BICUBIC)

def load_dataset(path):
    # normalization taken from https://discuss.pytorch.org/t/data-preprocessing-for-tiny-imagenet/27793
    upsample = BicubicUpsampling(256)
    crop_size = 224
    center_crop = transforms.CenterCrop(crop_size)
    rand_crop = transforms.RandomCrop(size=crop_size)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trfs_train = transforms.Compose([upsample, rand_crop, hflip, to_tensor, normalize])
    trfs_val = transforms.Compose([upsample, center_crop, to_tensor, normalize])
    train = ImageFolder(os.path.join(path, 'train'), transform=trfs_train)
    val = ImageFolder(os.path.join(path, 'val'), transform=trfs_val)
    return train, val

def count_parameters(model):
    """Counts the total number of model parameters."""
    return sum(params.numel() for params in model.parameters())