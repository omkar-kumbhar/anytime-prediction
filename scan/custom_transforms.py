import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import random

class AllRandomBlur(object):
    def __init__(self, kernel=7, std=0.9):
        self.kernel = kernel
        self.all_devs = np.arange(0.0,std+0.1,0.1)
    
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
    def __init__(self, mean=0., std=1., contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast
        self.all_devs = np.arange(0.0,std+0.01,0.01)
    
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


class Clip(object):
    def __init__(self, min_=0., max_=1.):
        self.min_ = min_
        self.max_ = max_
        
    def __call__(self, tensor):
        """
        Clamp values in given range
        """
        return torch.clamp(tensor, min=self.min_, max=self.max_)
    
class LowContrast(object):
    def __init__(self, contrast):
        self.contrast = contrast
        
    def __call__(self, tensor):
        """
        Clamp values in given range
        """
        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        return tensor