#!/usr/bin/env python
# coding: utf-8

# In[1]:


from osgeo import gdal
import numpy as np
import random
import os
import re
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans as kmeans

def shuffle_list(*ls, seeds =99):
  random.seed(seeds)
  l =list(zip(*ls))

  random.shuffle(l)
  return zip(*l)

def raster2array(raster, bnd,x,y,wndo):
    rr = np.zeros((wndo, wndo, bnd))
    newValue = 0
    for b in range(bnd):
        band = raster.GetRasterBand(b+1)
        rr[:,:,b] = band.ReadAsArray(x, y, wndo, wndo)
        noDataValue = band.GetNoDataValue()
    rr[rr == noDataValue] = newValue
    return rr.astype(np.uint8)

def raster2array(raster, bnd,x,y,wndox,wndoy):
    rr = np.zeros((wndoy, wndox, bnd))
    newValue = 0
    for b in range(bnd):
        band = raster.GetRasterBand(b+1)
        rr[:,:,b] = band.ReadAsArray(x, y, wndox, wndoy)
        noDataValue = band.GetNoDataValue()
    rr[rr == noDataValue] = newValue
    return rr.astype(np.uint8)

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
  
  
def get_all_images(folder, ext):

    all_files = []
    #Iterate through all files in folder
    for file in sorted_aphanumeric(os.listdir(folder)):
        #Get the file extension
        _,  file_ext = os.path.splitext(file)

        #If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    #Get list of all files
    return all_files

def fetch_all_tiles(ipath, image_id, hmargin, wmargin, steps,size = 256):
    
    
    raster = gdal.Open(ipath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    raster = None
    
    
    y_coord= (np.arange(hmargin,h+1,size))
    y_coord = y_coord[:-1]
    
    x_coord= np.arange(wmargin,w+1,size)
    x_coord = x_coord[:-1]
    

    i= (y_coord.shape[0]) 
    j = (x_coord.shape[0])
    num_tiles = i*j
    
    print("image: {}, number of tiles: {}".format(image_id,num_tiles))
    
    x_coord = x_coord*np.ones((i,j))
    y_coord = y_coord.reshape(i,1)*(np.ones((i,j)))
    
    tile_id = np.arange(0,num_tiles,1).reshape(i,j)
    img_id = np.ones_like(tile_id) * image_id 
    y_coord = np.around(y_coord)
    x_coord = np.around(x_coord)


    step = steps
    tile_id = list(tile_id.ravel()[::step])
    y_coord = list(y_coord.ravel()[::step])
    x_coord = list(x_coord.ravel()[::step])
    img_id = list(img_id.ravel()[::step])

    return img_id,tile_id,y_coord,x_coord

def apply_fetch_all_tiles(filepath,img_id,tile_id,y_coord,x_coord,steps, unique_id):
    
    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    
    raster = None

    img_id1,tile_id1,y_coord1,x_coord1 = fetch_all_tiles(filepath,unique_id,0,0,steps)     
    
    img_id.extend(img_id1)
    tile_id.extend(tile_id1)
    y_coord.extend(y_coord1)
    x_coord.extend(x_coord1)
    
    return img_id,tile_id,y_coord,x_coord



def fetch_tiles_at_random(start_w, start_h, w,h, wmargin, hmargin, starting_tile_id ,number_of_tiles, image_id):
    
    np.random.seed(440+image_id)
    y_coord = list(np.squeeze(np.random.randint(start_h,(h-hmargin),size=(number_of_tiles,1)).astype(np.float64)))
    np.random.seed(494+image_id)
    x_coord = list(np.squeeze(np.random.randint(start_w,(w-wmargin),size=(number_of_tiles,1)).astype(np.float64)))
    tile_id = list(np.squeeze(np.arange(starting_tile_id,starting_tile_id + number_of_tiles,1).reshape(number_of_tiles,1)))
    img_id = list(np.squeeze(np.zeros((number_of_tiles,1),dtype=np.uint8)+image_id))

    return img_id,tile_id,y_coord,x_coord

def apply_fetch_tiles_at_random(filepath,img_id,tile_id,y_coord,x_coord,starting_tile_id,number_of_tiles, unique_id,size = 256):
    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    
    raster = None
    try:
        img_id1,tile_id1,y_coord1,x_coord1 = fetch_tiles_at_random(0,0, w, h,size,size,starting_tile_id,number_of_tiles,unique_id)   
    except:
        print("skipping to select tiles at random")
    else:
        img_id.extend(img_id1)
        tile_id.extend(tile_id1)
        y_coord.extend(y_coord1)
        x_coord.extend(x_coord1)
   
    return img_id,tile_id,y_coord,x_coord


def iou(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum((y_true * y_pred), axis = (0,1,2))
    union = K.sum((y_true + y_pred), axis = (0,1,2)) - K.sum((y_true * y_pred), axis = (0,1,2))
    
    iu = ((intersection + smooth)/ (union + smooth))
    
    return K.mean(iu)
    

def iou_loss(y_true, y_pred,ncl = 1.0):
    return (ncl-iou(y_true, y_pred))


 
def dice(y_true, y_pred):
    smooth1 = 1e-6
    
    num1 = K.sum((y_true *  y_pred), axis = (0,1,2))
    dnm1 = K.sum((y_true +  y_pred), axis = (0,1,2))     
    
    f1 = ((2*num1 + smooth1)/ (dnm1 + smooth1)) 
    
    return K.sum(f1)
 
def dice_loss(y_true, y_pred,ncl = 4.0):
    return (ncl-dice(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred))

def generate_labels(image,ndvi,ndwi):
        img1 = process_kmeans2(image)
        img2 = np.zeros_like(img1)
        clt = norm_classes(img1,img2,ndvi,ndwi)
        return clt


def process_kmeans2(imgn, K=4):
     
    img1 = imgn.reshape((-1,imgn.shape[-1]))
    img1 = np.float32(img1)

    kmn = kmeans(n_clusters=4, n_init=30, max_iter=10, tol=0.01, precompute_distances='auto', verbose=0, random_state=36, copy_x=True, n_jobs=None, algorithm='auto').fit(img1)
    label = kmn.labels_
    return label.reshape((imgn.shape[0],imgn.shape[1]))

def norm_classes(img1,img2,ndvi,ndwi):
    for c in list(set(img1.flatten())):    
        mask = np.zeros_like(img1)
        mask[img1==c] =1
        mean_ndvi = np.sum(ndvi[img1==c])/np.sum(mask)
        mean_ndwi = np.sum(ndwi[img1==c])/np.sum(mask)
        if (mean_ndvi<0.2 and mean_ndvi>0 and mean_ndwi<0):
            img2[img1==c] = 7
        elif (mean_ndvi<0.4 and mean_ndvi>=0.2 and mean_ndwi<0):
            img2[img1==c] = 8
        elif (mean_ndvi>=0.4 and mean_ndwi<0):
            img2[img1==c] = 9
        elif (mean_ndvi<0 and mean_ndwi>=0.3):
            img2[img1==c] = 1 
        elif (mean_ndvi<0 and mean_ndwi<0.3 and mean_ndwi>0):
            img2[img1==c] = 2 
        elif (mean_ndvi<0 and mean_ndwi<0):
            img2[img1==c] = 3
        elif (mean_ndvi<=0.2 and mean_ndvi>0 and mean_ndwi<=0.3 and mean_ndwi>0):
            img2[img1==c] = 4 
        elif (mean_ndvi<=0.4 and mean_ndvi>0.2 and mean_ndwi<=0.3 and mean_ndwi>0):
            img2[img1==c] = 5 
        elif (mean_ndvi==0 and mean_ndwi==0):
            img2[img1==c] = 0
        else:
            img2[img1==c] = 6
    return img2

def add_veg_water_index(image):
    h,w = image.shape[:2]
    image2= np.zeros((h,w,6))
    image2[:,:,:4] = image
    del image
    r = image2[:,:,2]
    g = image2[:,:,1]
    nir = image2[:,:,3]

    num1 = nir-r
    dnm1 = nir+r
    num2= g-nir
    dnm2 = g+nir

    del nir,r,g
    ndvi = np.divide(num1,dnm1, where = (dnm1!=0), dtype = np.float64)
    ndwi = np.divide(num2,dnm2, where = (dnm2!=0), dtype = np.float64)

    del num1,num2,dnm1,dnm2
    ndvi[ndvi<0] = 0
    ndwi[ndwi<0] = 0

    image2[:,:,4] = ndvi*255
    image2[:,:,5] = ndwi*255

    return image2,ndvi,ndwi

def raster2arr(raster, bnd,x,y,wndo):
    rr = np.zeros((wndo, wndo, bnd))
    newValue = 0
    for b in range(bnd):
        band = raster.GetRasterBand(b+1)
        img = band.ReadAsArray(x, y, wndo, wndo)
#         noDataValue = band.GetNoDataValue()
        noDataValue = 255
        img[img == noDataValue] = newValue
        rr[:,:,b] = img
    return rr.astype(np.uint8)

def contrast(image_array: np.ndarray):
        v_min, v_max = np.percentile(image_array, (0.2, 99.8))
        return exposure.rescale_intensity(image_array, in_range=(v_min, v_max))

