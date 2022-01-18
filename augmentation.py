import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from torchvision.transforms import transforms

def get_augmentation():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]
    )

    valid_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]
    )

    return train_transforms, valid_transforms
    
