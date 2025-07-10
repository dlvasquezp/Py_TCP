# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:00:54 2024

@author: vasquezpinzondavid
"""
import matplotlib.pyplot as plt
import torch

import pathlib
pathlib.PosixPath = pathlib.WindowsPath
#model = torch.hub.load("C:/Users/adminKnorr/Documents/David/Github/PyLV_TCP_1.2/yolov5", "custom", path="C:/Users/adminKnorr/Documents/David/Github/PyLV_TCP_1.2/trainedModels/Macrof_Yolo5_UKJ.pt", source="local",force_reload=True)
#model = torch.hub.load("C:/Users/adminKnorr/Documents/David/Github/PyLV_TCP_1.2/yolov5", "custom", path="C:/Users/adminKnorr/Documents/David/Github/PyLV_TCP_1.2/trainedModels/best.pt", source="local",force_reload=True)
model = torch.hub.load("C:/Users/vasquezpinzondavid/Documents/GitHub/PyLV_TCP_1.2/yolov5", "custom", path="C:/Users/vasquezpinzondavid/Documents/GitHub/PyLV_TCP_1.2/trainedModels/best.pt", source="local",force_reload=True)

#im="C:/Users/adminKnorr/Documents/UKJ/Hemospec/PAR-033-1-WBC/2/Image/ImageFileName10.jpg"
im="temp/ImageFileName5.jpg"
original = plt.imread(im) 

results = model(im)
results.show()

#%%
masks_np = results.pandas().xyxy[0].sort_values("xmin")

confThrd = 0.45
minArea  = 19000 #px2
minBorDist = 50 #px

centers = []
areaList =[]
for idx,row in masks_np.iterrows():
    xMean = (row.xmax + row.xmin)/2
    yMean = (row.ymax + row.ymin)/2
    area  = abs((row.xmax-row.xmin)*(row.ymax-row.ymin))
    areaList.append(area)
    if xMean > minBorDist and yMean > minBorDist:
        if xMean < (original.shape[1]-minBorDist):
            if yMean < (original.shape[0]-minBorDist):
                if row.confidence > confThrd and area > minArea:
                    centers.append([xMean,yMean])

#%%
%matplotlib qt5
original = plt.imread(im) 
plt.figure()              
plt.imshow(original,cmap='grey')
for q in centers:
    plt.scatter(q[0],q[1],c='red')
plt.show()
