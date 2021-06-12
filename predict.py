#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from osgeo import gdal
import os

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import utils as utils
import train as train
    
def predict_on_image(model,batch,image,label,heights,widths,h,w,d,size=256):

    m=1
    for k in heights[0:-1]:
        p = 0
        arr = np.zeros(((len(widths)-1),size,size,d+2),dtype = np.float64)
        img = np.zeros((size,size,d))
        for n in widths[0:-1]:
            img = image[k:(k+size), n:(n+size),:]
            arr[p,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
            p+=1
        
        arr *= 1./255
        arr = model.predict(arr, batch_size = batch, verbose = 0)

        arr = np.argmax(arr,axis = -1)

        print("\r", m, '/',(len(heights)-1), end=" ")

        p = 0
        for n in widths[0:-1]:
            label[k:(k+size), n:(n+size)] = arr[p,:,:]
            p+=1
        m+=1
    
    return label;
    
def adjust_height(model,batch,image,label,heights,h,w,d,size=256):
    p = 0
    arr = np.zeros(((len(heights)-1),size,size,d+2),dtype = np.float64)
    
    for k in heights[0:-1]:    
        img = image[k:(k+size), w-size:w,:]
        arr[p,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
        p+=1
    
    arr *= 1./255
    arr = model.predict(arr, batch_size = batch, verbose = 0)
    
    arr = np.argmax(arr,axis = -1)
    print('adjusted height')
    
    p = 0
    for k in heights[0:-1]:
        label[k:(k+size), w-size:w] = arr[p,:,:]
        p+=1
    
    return label
    
def adjust_width(model,batch,image,label,widths,h,w,d,size=256):
    p = 0
    arr = np.zeros(((len(widths)-1),size,size,d+2),dtype = np.float64)
    
    for n in widths[0:-1]:
        img = image[h-size:h, n:(n+size),:]
        arr[p,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
        p+=1
    
    arr *= 1./255
    arr = model.predict(arr, batch_size = batch, verbose = 0)
    
    arr = np.argmax(arr,axis = -1)

    print('adjusted width')
    p = 0
    for n in widths[0:-1]:
        label[h-size:h, n:(n+size)] = arr[p,:,:]
        p+=1
    
    return label
    
def batch_predict(filepath,model,batch,hi=0,wi=0,size = 256):

    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    bands = raster.RasterCount
    image = (utils.raster2array(raster,bands,0,0,w,h)).astype(np.uint8)
    raster = None 
    
    
    h,w,d = image.shape
    
    heights = list(range(hi,h+1,size))
    widths = list(range(wi,w+1,size))

    lHeights = len(heights)
    lWidths = len(widths)

        
    label = np.zeros((h,w))
    print('predicting')
    
    if h != size and w != size:
        label = predict_on_image(model,batch,image,label,heights,widths,h,w,d,size)
    
        label = adjust_height(model,batch,image,label,heights,h,w,d,size)

        label = adjust_width(model,batch,image,label,widths,h,w,d,size)
    
    img = (image[h-size:h, w-size:w,:])  
    arr = np.zeros((1,size,size,d+2),dtype = np.float64)
    arr[0,:,:,:],ndvi,ndwi = utils.add_veg_water_index(img)
    
    arr *= 1./255
    arr = model.predict(arr, batch_size = 1, verbose = 0)
    
    label[h-size:h, w-size:w] = np.squeeze(np.argmax(arr,axis = -1))
    
    return label.astype(np.uint8)


def predict_all_images(flist,model,batch,inTest,size = 256):
    
    for f, fns in enumerate(flist):
        label1 = batch_predict(fns,model,batch)
        h,w = label1.shape[:2]
        print(label1.shape)

        print('0: {}, 1: {}, 2: {}, 3: {}'.format(np.sum((label1[label1==0])+1), np.sum((label1[label1==1])), np.sum((label1[label1==2]))//2, np.sum((label1[label1==3]))//3))

        
        if inTest is False:
            label2 = batch_predict(fns,model,batch,size//2,0)
            label3 = batch_predict(fns,model,batch,0,size//2)
            label4 = batch_predict(fns,model,batch,size//2,size//2)

            heights = list(range(0,h+1,size))
            widths = list(range(0,w+1,size))


            m=1
            for k in heights[0:-1]:
                label1[(k-size//4):(k+size//4),:] = label2[(k-size//4):(k+size//4),:]

            for n in widths[0:-1]:
                label1[:, (n-size//4):(n+size//4)] = label3[:, (n-size//4):(n+size//4)]

            for k in heights[0:-1]:
                for n in widths[0:-1]:
                    label1[(k-size//4):(k+size//4),(n-size//4):(n+size//4)] = label4[(k-size//4):(k+size//4),(n-size//4):(n+size//4)]


            print('0: {}, 1: {}, 2: {}, 3: {}'.format(np.sum((label1[label1==0])+1), np.sum((label1[label1==1])), np.sum((label1[label1==2]))//2, np.sum((label1[label1==3]))//3))

        cv.imwrite("./predictions/prediction_{}.tif".format(f+1), (label1).astype(np.uint8))              
        print('image {} saved'.format(f+1))