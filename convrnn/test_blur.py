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
import json
import os
import argparse

# tf.enable_v2_behavior()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from rcnn_sat import preprocess_image, bl_net
from load_data import load_dataset, load_dataset_h5, prep_pixels, prep_pixels_h5
from custom_transforms import add_gaussian_blur

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str)
parser.add_argument('--tag', default='blur-gray', type=str)
parser.add_argument('--color', default='gray', type=str)
parser.add_argument('--download-data', default=False, type=bool)
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

model.load_weights(args.load_path,skip_mismatch=True,by_name=True)

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

blur_results ={}
stds = [0.0,1.0,1.25,1.5,1.75,2.0,3.0]
start_time = time.strftime('%Y.%m.%d_%H.%M.%S')
for std in stds:
    trialX = np.zeros((len(testX),128,128,3))
    for i,image in enumerate(testX):
        trialX[i] = add_gaussian_blur(image, std=std)
    predictions = model.predict(trialX)
    blur_results[std] = [sum(np.argmax(predictions[i],axis=1)==np.argmax(testy[i],axis=1))/len(predictions[i]) for i in range(8)]
    with open(os.path.join('./model', '{}_results_{}.json'.format(
        args.tag,
        start_time
    )), 'w') as f:
        json.dump(blur_results, f)
