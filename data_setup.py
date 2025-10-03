
# data_setup.py
import torch.utils as utils
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    """Wrapper to make Albumentations compatible with PyTorch datasets"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL to numpy
        img = np.array(img)
        # Apply albumentations transform
        transformed = self.transform(image=img)
        return transformed['image']

class DataSetup:
    def __init__(self, batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False, num_workers=2, pin_memory=None, train_transforms=None, test_transforms=None):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms if train_transforms else self.get_train_transforms()
        self.test_transforms = test_transforms if test_transforms else self.get_test_transforms()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

    def get_train_transforms(self):
        """Albumentations transforms for training with required augmentations"""
        # CIFAR-10 mean: (0.4914, 0.4822, 0.4465) -> [125, 123, 114] for 0-255 range
        fill_value = [125, 123, 114]

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=fill_value,
                mask_fill_value=None,
                p=0.5
            ),
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
            ToTensorV2()
        ])
        return AlbumentationsTransform(train_transform)

    def get_test_transforms(self):
        """Albumentations transforms for testing (only normalization)"""
        test_transform = A.Compose([
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
            ToTensorV2()
        ])
        return AlbumentationsTransform(test_transform)

    def get_train_datasets(self):
        return datasets.CIFAR10('../data', train=True, download=True, transform=self.train_transforms)

    def get_test_datasets(self):
        return datasets.CIFAR10('../data', train=False, download=True, transform=self.test_transforms)

    def get_train_loader(self):
        train_dataset = self.get_train_datasets()
        return utils.data.DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=self.shuffle_train, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_test_loader(self):
        test_dataset = self.get_test_datasets()
        return utils.data.DataLoader(test_dataset, batch_size=self.batch_size_test, shuffle=self.shuffle_test, num_workers=self.num_workers, pin_memory=self.pin_memory)

