#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from osgeo import gdal
import os
import numpy as np

from time import time
import keras
import h5py
import random
from scipy import ndarray
from skimage import exposure
from random import random as rd
import cv2 as cv

from sklearn.model_selection import train_test_split
import re
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from tensorflow.keras.metrics import Accuracy,MeanIoU
from tensorflow.keras.utils import to_categorical
# from h2o4gpu.solvers.kmeans import KMeans as kmeans
import utils as utils


# In[ ]:


def img_generator(image_path,label_path,image_id,tile_id,y_coord,x_coord,batch):
    
    
    def com_gen(image,label):
        K.set_image_data_format('channels_last')
        def combine_generator(gen1, gen2):
            while True:
                yield(gen1.next(), gen2.next())
        
        data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=45,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         #brightness_range=[0.8,1.2],
                         #shear_range=0.2,
                         #channel_shift_range = 20,
                         zoom_range=0.2,
                         fill_mode='nearest',
                         data_format = "channels_last",
                         rescale = 1./255)
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        #image_datagen.fit(image, augment=True, rounds=1, seed=42)
        #mask_datagen.fit(label, augment=True, rounds=1, seed=42)
        image_generator = image_datagen.flow(image, batch_size=1, seed=42)
        mask_generator = mask_datagen.flow(label, batch_size=1, seed=42)
        train_generator = combine_generator(image_generator,mask_generator)
        return train_generator

    
    total_images = len(y_coord)
    all_images_id =list(range(total_images))
    random.seed(948)
    random.shuffle(all_images_id)
    
    current_image_id = 0
    while (current_image_id<total_images):
        
        batch_input = np.zeros((batch,size,size,6))
        batch_output = np.zeros((batch,size,size,4))
        
        
        for batch_index in range(batch):
            index = all_images_id[current_image_id]
            x = x_coord[index]
            y = y_coord[index]
            img_id = image_id[index]
            
        
            raster = gdal.Open(image_path[img_id])
            img = utils.raster2array(raster,4,x,y,size,size)
            raster = None
            batch_input[batch_index,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
     
            raster = gdal.Open(label_path[img_id])
            clt = utils.raster2array(raster,1,x,y,size,size)
            raster = None
            batch_output[batch_index,:,:,:] =  255.*to_categorical(clt,num_classes=4)
            del img,clt
            current_image_id += 1
            
        return (com_gen(batch_input, batch_output))
    
def msk_generator(image_path,label_path,image_id,tile_id,y_coord,x_coord,batch):    
    
    
    def com_gen(image2,label2):
        K.set_image_data_format('channels_last')
        def combine_generator(gen1, gen2):
            while True:
                yield(gen1.next(), gen2.next())                                   
  
        data_gen_args2 = dict(data_format = "channels_last",
                             rescale = 1./255)
        image_datagen2 = ImageDataGenerator(**data_gen_args2)
        mask_datagen2 = ImageDataGenerator(**data_gen_args2)
        #image_datagen.fit(image, augment=True, rounds=5, seed=42)
        #mask_datagen.fit(label, augment=True, rounds=5, seed=42)
        image_generator2 = image_datagen2.flow(image2, batch_size=1, seed=56)
        mask_generator2 = mask_datagen2.flow(label2, batch_size=1, seed=56)
        val_generator = combine_generator(image_generator2,mask_generator2)
        return val_generator
    
    total_images = len(y_coord)
    all_images_id =list(range(total_images))
    random.seed(948)
    random.shuffle(all_images_id)
    
    current_image_id = 0
    while (current_image_id<total_images):
        
        batch_input = np.zeros((batch,size,size,6))
        batch_output = np.zeros((batch,size,size,4))
        
        
        for batch_index in range(batch):
            index = all_images_id[current_image_id]
            x = x_coord[index]
            y = y_coord[index]
            img_id = image_id[index]
            
        
            raster = gdal.Open(image_path[img_id])

            img = utils.raster2array(raster,4,x,y,size,size)
            raster = None           
            batch_input[batch_index,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
            
            raster = gdal.Open(label_path[img_id])
            clt = utils.raster2array(raster,1,x,y,size,size)
            raster =None
            batch_output[batch_index,:,:,:] = 255.*to_categorical(clt,num_classes=4)
            del img,clt
            current_image_id += 1
        
        return (com_gen(batch_input, batch_output))


# In[ ]:


size = 256
def unet(size, lri, input_height = size, input_width = size, nClasses = 4):

    input_size = (input_width, input_height, 6)
    input1 = Input(shape = input_size)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input1)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(nClasses, (1,1), activation = 'softmax',padding = 'same', kernel_initializer = 'he_normal')(conv9)

    model = Model(inputs = input1, outputs = conv10)

    model.compile(optimizer = Adam(lr  = lri), loss= utils.dice_loss, metrics = [utils.iou,utils.dice])

    return model




# In[ ]:


def fetch_tiles_info(ipath):
    img_id  = []
    tile_id = []
    y_coord = []
    x_coord = []


    for i in range(len(ipath)):
        img_id,tile_id,y_coord,x_coord = utils.apply_fetch_all_tiles(ipath[i],img_id,tile_id,y_coord,x_coord,1,i)
        
        img_id,tile_id,y_coord,x_coord = utils.apply_fetch_tiles_at_random(ipath[i],img_id,tile_id,y_coord,x_coord,len(img_id),len(img_id),i)
    
    return img_id,tile_id,y_coord,x_coord


# In[ ]:





# In[ ]:


def training(train_generator,val_generator,steps_train,steps_val,lri = 1e-3,epoch =50,size=256,seed = 100):
    K.clear_session()
    model = 0
    del model         
    K.set_image_data_format('channels_last')
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    

    model = unet(size, lri)
    model.summary()
    
    checkpoint = ModelCheckpoint("./model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
    history = model.fit(train_generator, steps_per_epoch = steps_train, epochs = epoch ,verbose = 2, callbacks = [checkpoint, early], validation_data= val_generator, validation_steps = steps_val)    


# In[ ]:




