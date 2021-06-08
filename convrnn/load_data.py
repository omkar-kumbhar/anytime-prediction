from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import h5py
import os

def load_dataset():
    """download dataset from keras datasets
    """
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    # rcnn requires 8 arrays of targets. 
    
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    """ resize, normalize image
    """
    # convert from integers to floats
    data_train = np.zeros((len(train),128,128,3))
    data_test = np.zeros((len(test),128,128,3))
    for i,image in enumerate(train):
        #im = np.transpose(image,(1,2,0))
        large_im = cv2.resize(image,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
        
        data_train[i] = large_im
    for i,image in enumerate(test):
        #im = np.transpose(image,(1,2,0))
        large_im = cv2.resize(image,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
        data_test[i] = large_im
    
    del train
    del test

    train_norm = data_train.astype('float32')
    test_norm = data_test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    # return train_norm, test_norm
    return train_norm,test_norm

# trainX, trainY, testX, testY = load_dataset()
# trainX, testX = prep_pixels(trainX, testX)

def prep_pixels_h5(trainX,trainy, testX,testy, data_root, color):
    """prep pixels and save
    """
    # convert from integers to floats
    # data_train = np.zeros((len(trainX),128,128,3))
    # data_test = np.zeros((len(testX),128,128,3))

    trainX_path = os.path.join(data_root, 'input_data.h5')
    trainy_path = os.path.join(data_root, 'labels.h5')
    # testX_path = './data/testX.h5'
    # testy_path = './data/testy.h5'
    
    with h5py.File(trainy_path, 'w') as f:
        # create dataset for labels
        f.create_dataset("trainy",(8,len(trainy),10),np.uint8)
        f.create_dataset("testy",(8,len(testy),10),np.uint8)
        f["trainy"][...] = np.array([trainy for i in range(8)])
        f["testy"][...] = np.array([testy for i in range(8)])

    with h5py.File(trainX_path, 'a') as f:
        # create h5py dataset for arrays
        f.create_dataset("trainX",(len(trainX),128,128,3),np.float32)
        f.create_dataset("testX",(len(testX),128,128,3),np.float32)
        
        for i,image in enumerate(trainX):
            #im = np.transpose(image,(1,2,0))
            large_im = cv2.resize(image,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
            if color == 'gray':
                large_im = cv2.cvtColor(np.array(large_im,dtype=np.uint8), cv2.COLOR_BGR2GRAY)
                large_im = np.stack([large_im for i in range(3)],axis=2)
            # data_train[i] = large_im
            f["trainX"][i,...] = large_im.astype('float32') / 255.0
        for i,image in enumerate(testX):
            #im = np.transpose(image,(1,2,0))
            large_im = cv2.resize(image,dsize=(128,128),interpolation=cv2.INTER_CUBIC)
            if color == 'gray':
                large_im = cv2.cvtColor(np.array(large_im,dtype=np.uint8), cv2.COLOR_BGR2GRAY)
                large_im = np.stack([large_im for i in range(3)],axis=2)
            # data_test[i] = large_im
            f["testX"][i,...] = large_im.astype('float32') / 255.0

# trainX, trainY, testX, testY = load_dataset()
# os.makedirs('./data', exist_ok = True)
# prep_pixels_h5(trainX, trainY, testX, testY)

def load_dataset_h5(data_root):
    X = os.path.join(data_root, 'input_data.h5')
    y = os.path.join(data_root, 'labels.h5')

    with h5py.File(X,'r') as f:
        trainX = f["trainX"][:]
        testX = f["testX"][:]
    with h5py.File(y,'r') as f:
        trainy = f["trainy"][:]
        testy = f["testy"][:]

    return trainX,trainy,testX,testy