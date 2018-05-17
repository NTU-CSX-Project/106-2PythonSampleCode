
# coding: utf-8

# In[1]:

import os
os.environ['KERAS_BACKEND']='theano'
from PIL import Image
import numpy as np
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import random
np.random.seed(1024) 


# In[2]:

def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")

    imgs = os.listdir("/home/keras/Desktop/keras/mnist")
    num = len(imgs)  # 42000
    for i in range(num):
        img = Image.open("/home/keras/Desktop/keras/mnist/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label


# In[3]:

X_data, Y_data  = load_data()
# 打亂資料
index = [i for i in range(len(X_data))]
random.shuffle(index)
X_data = X_data[index]
Y_data = Y_data[index]


# In[4]:

X_train, X_test = X_data[:30000,:,:,:], X_data[30000:,:,:,:]
Y_train, Y_test = Y_data[:30000,], Y_data[30000:,]


# In[5]:

# input image dimensions
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1 )
input_shape = (28, 28, 1)


# In[6]:

Y_train = np_utils.to_categorical(Y_train, 10)


# In[7]:

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1 )
Y_test = np_utils.to_categorical(Y_test, 10)


# In[8]:

model = Sequential()
model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape = input_shape)) 
model.add(Activation('tanh'))
model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16, 3, 3, border_mode='valid')) 
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('tanh'))
model.add(Dense(10, init='normal'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])


# In[11]:

model.fit(X_train, Y_train, batch_size=100, nb_epoch=10,shuffle=True,verbose=1)


# In[9]:

score = model.evaluate(X_test, Y_test)


# In[10]:

print("Total Loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])


# In[ ]:



