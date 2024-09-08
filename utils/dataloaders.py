import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader


# Preprocessing pipelines
def custom_loader(path):
    return Image.open(path)

def get_trainDataLoader(batchsize):
    train_dir = '../data/train/'

    transform = v2.Compose([
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop([224]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.DatasetFolder(
        root=train_dir,
        loader=custom_loader,
        extensions=('jpg', 'jpeg', 'png'),
        transform=transform
    )

    loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)

    return loader

def get_validDataLoader(batchsize):
    valid_dir = '../data/valid/'

    transform = v2.Compose([
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop([224]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = datasets.DatasetFolder(
        root=valid_dir,
        loader=custom_loader,
        extensions=('jpg', 'jpeg', 'png'),
        transform=transform
    )

    loader = DataLoader(test_set, batch_size=batchsize, shuffle=True)

    return loader
