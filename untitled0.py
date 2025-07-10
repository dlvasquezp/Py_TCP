# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:51:09 2025

@author: vasquezpinzondavid
"""
import os

print(__file__)

print(os.path.dirname(__file__))

print(os.path.dirname(__file__)+'\\temp\\temp.tiff')

import socket
import numpy as np
import matplotlib.pyplot as plt
import select
import time
import os
###########################################################
import torch
import pathlib
import matplotlib.image
import cv2
from skimage import color
from skimage import io
print('LOADING MODEL...')
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load( os.path.dirname(__file__)+"/yolov5", "custom", path= os.path.dirname(__file__)+"/trainedModels/best.pt", source="local",force_reload=True)


def processImage(model,path):
    #original = io.imread(path)
    results = model(path)
    results.show()

    masks_np = results.pandas().xyxy[0].sort_values("ymin")

    confThrd = 0.4
    minArea  = 5000 #px2
    minBorDist = 40 #px

    centers = []
    areaList =[]
    nx, ny = original.shape[:2]
    x = np.linspace(0,ny-1,ny)
    y = np.linspace(0,nx-1,nx)
    xv, yv = np.meshgrid(x,y)
    segment= np.zeros((nx,ny),dtype=np.uint8)
    count  = 1
    for idx,row in masks_np.iterrows():
        xMean = (row.xmax + row.xmin)/2
        yMean = (row.ymax + row.ymin)/2
        area  = abs((row.xmax-row.xmin)*(row.ymax-row.ymin))
        areaList.append(area)
        if xMean > minBorDist and yMean > minBorDist:
            if xMean < (original.shape[1]-minBorDist):
                if yMean < (original.shape[0]-minBorDist):
                    if row.confidence > confThrd and area > minArea:
                        centers.append([xMean,yMean,row.xmin])
                        segment[np.where(((xv>row.xmin)*1 + (xv<row.xmax)*1 + (yv>row.ymin)*1 + (yv<row.ymax)*1)==4)]=(count%255)
                        count += 1
                        
    if centers==[]:
        centers=[0,0,0]
    
    return segment, centers


im="temp/ImageFileName3.jpg"
original = plt.imread(im) 

ySize,xSize=np.shape(original)
image = original

results = processImage(model,image)