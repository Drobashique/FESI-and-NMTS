from PIL import Image
import cv2 as cv
import skimage
from skimage import data, io, filters,morphology,segmentation,feature
from scipy import ndimage
from skimage import color
import numpy as np
import math
import sys, getopt

def rgb2gs(fd):
    newobj  = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range (fd.shape[0]):
        for j in range (fd.shape[1]):
            newobj[i,j] = 0.2989*fd[i,j,0] + 0.5870 * fd[i,j,1] + 0.1140 * fd[i,j,2]
    return newobj

def tr(fd,b,c):
    newobj  = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if fd[i,j] > b:
                newobj[i,j] = c
            else:
                newobj[i,j] = 255

    return newobj
def vid(fd):
    newobj = np.zeros((fd.shape[0],fd.shape[1],3))
    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if fd[i,j] == 200:
                newobj[i,j,0] = 0
                newobj[i,j,1] = 127
                newobj[i,j,2] = 0
            else:
                newobj[i,j,:] = fd[i,j]
    return newobj
    
def tr_inv(fd,b,c):
    newobj  = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if fd[i,j] > b:
                newobj[i,j] = 0
            else:
                newobj[i,j] = c
    return newobj

def tr12(fd,b):
    newobj  = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if fd[i,j] > b:
                newobj[i,j] = 255
            else:
                newobj[i,j] = 0
    return newobj

def am(fd):
    max = 0
    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if fd[i,j] > max:
                max = fd[i,j]
                im = i
                jm = j
    return (im,jm)
    

def fesi(fd):
    data = fd.astype(np.uint8) 
    omg = Image.fromarray(data)
    omg.show()
    #laplace = abs(skimage.filters.laplace(fd))
    laplace = skimage.feature.canny(fd)
    return laplace
    blurred = skimage.filters.gaussian(laplace,multichannel=False)
    mask = tr(blurred,blurred.mean(),150)
    temp = tr_inv(blurred,blurred.mean(),255)
    dseed = ndimage.distance_transform_edt(temp)
    mask = skimage.filters.median(mask,morphology.disk(7)) + 100 
    mask = tr12(mask,mask.min())

    mask = morphology.binary_opening(mask,morphology.disk(7))

    newobj = np.zeros((fd.shape[0],fd.shape[1]))
    t1 = am(dseed)[0]
    t2 = am(dseed)[1]
    tup = (t1,t2)
    print(mask[tup])

    for i in range (0,fd.shape[0]):
        for j in range (fd.shape[1]):
            if mask[i,j] == 1:
                newobj[i,j] = 0
            else:
                newobj[i,j] = 255
    mask = newobj
    mask = skimage.segmentation.flood_fill(mask, tup, 0)
    distance = ndimage.distance_transform_edt(mask)
    final_mask = mask.copy()
    dmax = distance.max()
    globalMax = distance.max()
    seeds = []
    for i in range(1):
        start = (am(distance)[0],am(distance)[1])
        s1 = distance.argmax()
        seeds.append((distance.argmax(), dmax))
        mask = skimage.segmentation.flood_fill(mask, start, 0)
        distance[mask == 0] = 0
        dmax = distance.max()
        if (dmax > 0.6*globalMax) or np.isclose(seeds, s1).any():
            #print('123')
            final_mask = skimage.segmentation.flood_fill(final_mask, start, 200)
        #t = final_mask
        #t = t.astype(np.uint8) 
        #omg = Image.fromarray(t)
        #omg.show()
        final_mask[final_mask != 200] = 0
        final_mask[final_mask == 200] = 255
        #t = final_mask
        #t = t.astype(np.uint8) 
        #omg = Image.fromarray(t)
        #omg.show()
        print(dmax)
    return final_mask
    
im = Image.open("/Users/vladimirlisovoi/desktop/testim/s1_test_04.png")
data = np.array(im)
fd = data.astype(np.float)
fd = rgb2gs(fd)
fd = fesi(fd)
#fd = vid(fd)
fd = fd.astype(np.uint8) 
omg = Image.fromarray(fd)
omg.show()
omg.save("/Users/vladimirlisovoi/desktop/testim/out.png")
