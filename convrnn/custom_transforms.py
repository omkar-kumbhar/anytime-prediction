import numpy as np
from torchvision import transforms
import torch

def all_random_blur(image, kernel=7, std=0.9):
    all_devs = np.arange(0.0, std+0.1, 0.1)
    std = np.random.choice(all_devs)

    if std != 0.0:
        image = np.transpose(image, (2,0,1))
        image = transforms.functional.gaussian_blur(torch.Tensor(image), kernel, sigma=std)
        image = np.transpose(image.numpy(), (1,2,0))
    
    return image

def add_gaussian_blur(image, kernel=7, std=0.9):
    if std != 0.0:
        image = np.transpose(image, (2,0,1))
        image = transforms.functional.gaussian_blur(torch.Tensor(image), kernel, sigma=std)
        image = np.transpose(image.numpy(), (1,2,0))
    
    return image

def all_random_noise(image, std=0.04, mean=0, contrast=0.1):
    std = np.random.choice(np.arange(0.0, std+0.01, 0.01))
    noise = np.array([])
    n = image.shape[0] * image.shape[1]
    sd2 = std * 2

    while len(noise) < n:
        # more samples than we require
        m = 2 * (n - len(noise))
        new = np.random.randn(m) * std

        # remove out-of-range samples
        new = new[new >= -sd2]
        new = new[new <= sd2]

        # append to noise tensor
        noise = np.concatenate((noise, new))
    
    # pick first n samples and reshape to 2D
    noise = np.reshape(noise[:n], (image.shape[0], image.shape[1]))

    # stack noise and translate by mean to produce std + 
    newnoise = np.stack([noise, noise, noise], axis=2) + mean

    # shift image hist to mean = 0.5
    image = image + (0.5 - np.mean(image))

    # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
    # print(self.contrast)

    image = np.transpose(image, (2,0,1))
    image = transforms.functional.adjust_contrast(torch.Tensor(image), contrast)
    image = np.transpose(image.numpy(), (1,2,0))
    
    return image + newnoise + mean

def add_gaussian_noise(image, std=0.04, mean=0, contrast=0.1):
    noise = np.array([])
    n = image.shape[0] * image.shape[1]
    sd2 = std * 2

    while len(noise) < n:
        # more samples than we require
        m = 2 * (n - len(noise))
        new = np.random.randn(m) * std

        # remove out-of-range samples
        new = new[new >= -sd2]
        new = new[new <= sd2]

        # append to noise tensor
        noise = np.concatenate((noise, new))
    
    # pick first n samples and reshape to 2D
    noise = np.reshape(noise[:n], (image.shape[0], image.shape[1]))

    # stack noise and translate by mean to produce std + 
    newnoise = np.stack([noise, noise, noise], axis=2) + mean

    # shift image hist to mean = 0.5
    image = image + (0.5 - np.mean(image))

    # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
    # print(self.contrast)

    image = np.transpose(image, (2,0,1))
    image = transforms.functional.adjust_contrast(torch.Tensor(image), contrast)
    image = np.transpose(image.numpy(), (1,2,0))
    
    return image + newnoise + mean