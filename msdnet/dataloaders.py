import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import random

def get_test_dataloaders(args):
    """
    This code will add white noise(gaussian) to CIFAR-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None

    if args.mode == 'noise':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianNoise(0.,args.noise)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AddGaussianNoise(0.,args.noise)
                                ]))

    if args.mode == 'blur':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AddGaussianBlur(7, args.blur)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AddGaussianBlur(7, args.blur)
                                ]))

    if args.mode == 'gray':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                ]))
    elif args.mode == 'color':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader


def get_train_dataloaders(args):
    """
    This code will add white noise(gaussian) to CIFAR-10 images with 50% probability
    
    """
    train_loader, val_loader, test_loader = None, None, None
    # normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
    #                                  std=[0.2471, 0.2435, 0.2616])
    if args.mode == 'noise':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomNoise()
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomNoise()
                                ]))

    if args.mode == 'blur':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        AllRandomBlur(7)
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    AllRandomBlur(7)
                                ]))

    if args.mode == 'gray':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                ]))
    elif args.mode == 'color':
        train_set = datasets.CIFAR10(args.data_root, train=True,download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader

class AllRandomBlur(object):
    def __init__(self, kernel=7):
        self.kernel = kernel
        self.all_devs = np.arange(0.0,1.0,0.1)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoise(object):
    """
    Author: Omkar Kumbhar
    Description:
    Adding gaussian noise to images in the batch
    """
    def __init__(self, mean=0., std=1., contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast

    def __call__(self, tensor):
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AllRandomNoise(object):
    def __init__(self, mean=0., std=0.04, contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast
        self.all_devs = np.arange(0.0,0.05,0.01)
    
    def __call__(self, tensor):
        self.std = np.random.choice(self.all_devs)
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Apply Gaussian noise if a number between 1 and 10 is less or equal than 5
        """
        if random.randint(1,10) <= 5:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(object):
    def __init__(self, min_=0., max_=1.):
        self.min_ = min_
        self.max_ = max_
        
    def __call__(self, tensor):
        """
        Clamp values in given range
        """
        return torch.clamp(tensor, min=self.min_, max=self.max_)
    