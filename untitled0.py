# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:00:30 2025

@author: vasquezpinzondavid
"""
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
import json
print('LOADING MODEL...')
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load( os.path.dirname(__file__)+"/yolov5", "custom", path= os.path.dirname(__file__)+"/trainedModels/best.pt", source="local",force_reload=True)

modelInfo = {
    "model"     : "Yolo5_01",
    "confThrd"  : 0.51      ,
    "minArea"   : 5000      ,
    "minBorDist": 40        ,
    "resolution": 120
}


def processImage(model,image2, modelInfo, cellsBorder=True):
    original = image2 
    results  = model(image2)
    #results.show()

    masks_np = results.pandas().xyxy[0].sort_values("ymin")

    centers = []
    ny, nx = original.shape
    x = np.linspace(0,nx-1,nx)
    y = np.linspace(0,ny-1,ny)
    xv, yv = np.meshgrid(x,y)
    segment= np.zeros((ny,nx),dtype=np.uint8)
    
    masks_np['xMean'] = (masks_np.xmax + masks_np.xmin)/2
    masks_np['yMean'] = (masks_np.ymax + masks_np.ymin)/2
    masks_np['area']  = abs((masks_np.xmax-masks_np.xmin)*(masks_np.ymax-masks_np.ymin))
    
    ########### Confidence threshold ################
    if "confThrd" in modelInfo:
        masks_np = masks_np.drop(masks_np[masks_np.confidence<modelInfo["confThrd"]].index)
    ########### Area threshold ######################
    if "minArea" in modelInfo:
        masks_np = masks_np.drop(masks_np[masks_np.area<modelInfo["minArea"]].index)
    ########### Min border distance #################
    if "minBorDist" in modelInfo:
        masks_np = masks_np.drop(masks_np[masks_np.xMean<(modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.yMean<(modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.xMean>(nx-modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.yMean>(ny-modelInfo["minBorDist"])].index)
    ########### Min distance between cells ##########
    
    ########### Fill segmented image ################
    count  = 1
    for idx,row in masks_np.iterrows():
        centers.append([row.xMean,row.yMean,row.xmin])
        segment[np.where(((xv>row.xmin)*1 + (xv<row.xmax)*1 + (yv>row.ymin)*1 + (yv<row.ymax)*1)==4)]=(count%255)
        count += 1
    
                        
    if centers==[]:
        centers=[0,0,0]
    
    return segment, centers



modelInfo = {
    "model"     : "Yolo5_01",
    "confThrd"  : 0.51      ,
    "minArea"   : 5000      ,
    "minBorDist": 40        ,
    "resolution": 120
}

im="temp/ImageFileName1.jpg"
image2 = plt.imread(im)

segment, centers = processImage(model,image2, modelInfo, cellsBorder=True)

plt.figure()
plt.imshow(segment)
plt.show()

