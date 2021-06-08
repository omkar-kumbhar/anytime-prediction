import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from PIL import Image, ImageEnhance
import cv2
import urllib
import numpy as np
from tensorflow.keras.utils import to_categorical
import glob
from random import shuffle
import h5py
import torch
from torchvision import transforms
import math
import time
import os
import argparse

# tf.enable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from rcnn_sat import preprocess_image, bl_net
from load_data import load_dataset, load_dataset_h5, prep_pixels, prep_pixels_h5

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='color-gray', type=str)
parser.add_argument('--color', default='gray', type=str)
parser.add_argument('--download-data', default=False, type=bool)
parser.add_argument('--pretrained', default=True, type=bool)
args = parser.parse_args()
print(args)

data_root = '../data/{}'.format(args.color)
if args.download_data == True:
    trainX, trainy, testX, testy = load_dataset()
    os.makedirs(data_root, exist_ok = True)
    prep_pixels_h5(trainX, trainy, testX, testy, data_root, args.color)
    args.download_data = False

if args.download_data == False:
    trainX,trainy,testX,testy = load_dataset_h5(data_root)

input_layer = tf.keras.layers.Input((128, 128, 3))
model = bl_net(input_layer, classes=10, cumulative_readout=False)

if args.pretrained:
    model.load_weights('bl_imagenet.h5',skip_mismatch=True,by_name=True)

## Lets try fine tuning it
# tf.keras.utils.plot_model(model,to_file='check.png')

skip_layers = ['ReadoutDense','Sotfmax_Time_0','Sotfmax_Time_1',
              'Sotfmax_Time_2','Sotfmax_Time_3','Sotfmax_Time_4',
              'Sotfmax_Time_5','Sotfmax_Time_6','Sotfmax_Time_7']

for layer in model.layers:
  if layer.name in skip_layers:
    layer.trainable = True
  else:
    layer.trainable = False

# compile model with optimizer and loss
"""
B, BL and parameter-matched controls (B-K, B-F and B-D) were trained for a total of 90 epochs 
with a batch size of 100. B-U was trained using the same procedure but with a batch size of 64 
due to its substantially larger number of parameters.

The cross-entropy between the softmax of the network category readout and the labels 
was used as the training loss. For networks with multiple readouts (BL and B-U), 
we calculate the cross-entropy at each readout and average this across readouts. 
Adam [64] was used for optimisation with a learning rate of 0.005 and epsilon parameter 0.1. 
L2-regularisation was applied throughout training with a coefficient of 10−6.

"""
cce = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("pretrained_mp_{}.hdf5".format(args.tag), monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()

# trainy = np.transpose(trainy, (1,2,0))
# testy = np.transpose(testy, (1,2,0))
print(trainX.shape)
print(trainy.shape)

history = model.fit(x=datagen.flow(trainX, trainy[0],batch_size=32),
                    validation_data=(testX,testy[0]),
                    steps_per_epoch=len(trainX)//32,
                    epochs=100,callbacks=[checkpoint])

model.save('./model/{}_{}'.format(
    args.tag,
    time.strftime('%Y.%m.%d_%H.%M.%S'),
))
