import socket
import numpy as np
import matplotlib.pyplot as plt
import select
import time
import os
import numbers
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

def processImage(model,image2, modelInfo):
    original = image2 
    results  = model(image2)
    #results.show()

    masks_np = results.pandas().xyxy[0].sort_values("ymin")

    centers = []
    ny, nx = original.shape
    x = np.linspace(0,nx-1,nx)
    y = np.linspace(0,ny-1,ny)
    xv, yv = np.meshgrid(x,y)
    
    masks_np['xMean'] = (masks_np.xmax + masks_np.xmin)/2
    masks_np['yMean'] = (masks_np.ymax + masks_np.ymin)/2
    masks_np['area']  = abs((masks_np.xmax-masks_np.xmin)*(masks_np.ymax-masks_np.ymin))
    
    ########### Confidence threshold ################
    if "confThrd" in modelInfo and isinstance(modelInfo["confThrd"], numbers.Number):
        masks_np = masks_np.drop(masks_np[masks_np.confidence<modelInfo["confThrd"]].index)
    ########### Area threshold ######################
    if "minArea" in modelInfo and isinstance(modelInfo["minArea"], numbers.Number):
        masks_np = masks_np.drop(masks_np[masks_np.area<modelInfo["minArea"]].index)
    ########### Min border distance #################
    if "minBorDist" in modelInfo and isinstance(modelInfo["minBorDist"], numbers.Number):
        masks_np = masks_np.drop(masks_np[masks_np.xMean<(modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.yMean<(modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.xMean>(nx-modelInfo["minBorDist"])].index)
        masks_np = masks_np.drop(masks_np[masks_np.yMean>(ny-modelInfo["minBorDist"])].index)
    ########### Max overlap between cells ##########
    if ("maxOverlap" in modelInfo) and isinstance(modelInfo["maxOverlap"], numbers.Number) and ("maxIoU" in modelInfo) and isinstance(modelInfo["maxIoU"], numbers.Number):
        segmentList = []
        for idx,row in masks_np.iterrows():
            segment= np.zeros((ny,nx),dtype=np.uint8)
            segment[np.where(((xv>row.xmin)*1 + (xv<row.xmax)*1 + (yv>row.ymin)*1 + (yv<row.ymax)*1)==4)]= True
            segmentList.append(segment)
    
        checkList = []    
        for i in range(len(segmentList)):
            for j in range(i,len(segmentList)):
                if i != j:
                    segA = segmentList[i]
                    segB = segmentList[j]
                    intersection =  np.sum(segA+segB==2)
                    
                    if intersection > 0:
                        overlapA = intersection/np.sum(segA>0)
                        overlapB = intersection/np.sum(segB>0)
                        IoU = intersection/np.sum(segA + segB>0)
                        if IoU > modelInfo["maxIoU"] or overlapA>modelInfo["maxOverlap"] or overlapB>modelInfo["maxOverlap"]:
                            print(i,j,IoU)
                            checkList.append([i,j])
    
        dropList=[]
        for [i,j] in checkList:
            confA = masks_np.confidence.iloc[i]
            confB = masks_np.confidence.iloc[j]
            if confA > confB:
                dropList.append(masks_np.index[j])
            else:
                dropList.append(masks_np.index[i])
        masks_np.drop(dropList,inplace=True)
    
    ########### Fill segmented image ################
    segment= np.zeros((ny,nx),dtype=np.uint8)
    count  = 1
    for idx,row in masks_np.iterrows():
        centers.append([row.xMean,row.yMean,row.area])
        segment[np.where(((xv>row.xmin)*1 + (xv<row.xmax)*1 + (yv>row.ymin)*1 + (yv<row.ymax)*1)==4)]=(count%255)
        count += 1
                        
    if centers==[]:
        centers=[0,0,0]
    
    return segment, centers
    
##########################################################
def empty_socket(sock, limit=100):
    """remove the data present on the socket"""
    input = [sock]
    counter=0
    while 1:
        inputready, o, e = select.select(input,[],[], 1.0)
        if len(inputready)==0: break
        for s in inputready: s.recv(1)
        counter +=1
        if counter > limit: break

# Creating a socket instance
server_object = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

# Connecting to the localhost
ip_address = '127.0.0.1'
port = 5555

server_object.bind((ip_address, port))
server_object.listen()

############### IDLE #####################################
while True:
    print('PYTHON SERVER READY')
    conn, clientAddress = server_object.accept() # connection,address
    print(clientAddress)
    conn.settimeout(10)
    try:
        ############### Receive Image #####################################
        print("SERVER CONNECTED TO CLIENT")
        conn.send(b"SYN")
        message = conn.recv(7)
        if message.decode('utf-8') == "SYN+ACK":
            print("Receiving data...")
            ySize = int.from_bytes(conn.recv(4), 'big')
            xSize = int.from_bytes(conn.recv(4), 'big')
            data_receive = conn.recv(xSize*ySize)
            if len(data_receive) == 0:
                raise ConnectionResetError
            if len(data_receive)==(xSize*ySize):
                conn.send(b"ACK") 
                print("Done, transmition length: ",xSize,ySize,len(data_receive))
            else:
                print("Comunication error")
                raise ConnectionResetError
                
        ############### Receive Model Information ########################
        conn.send(b"SYN")
        message = conn.recv(7)
        if message.decode('utf-8') == "SYN+ACK":
            print("Receiving data...")
            infoLen   = int.from_bytes(conn.recv(4), 'big')
            infoBytes = conn.recv(infoLen)
            if len(infoBytes) == 0:
                raise ConnectionResetError
            if len(infoBytes)==(infoLen):
                conn.send(b"ACK") 
                print("Done, Information received")
            else:
                print("Comunication error")
                raise ConnectionResetError

        modelInfo = json.loads(infoBytes)
        print(modelInfo)
        
        ############### Process image #####################################
        print("Processing...")
        startTime = time.time()
        
        image=[]
        for q in range(ySize):
            yColumn = []
            for char in data_receive[int(q*xSize):int((q+1)*xSize)]:
                yColumn.append(char)
            image.append(yColumn)
        
        image2 = np.array(image,dtype=np.uint8)     
        
        try:
            segImage,maskCenters  = processImage(model,image2,modelInfo)
        except:
            print('########## Model Error ################')
            segImage,maskCenters  = processImage(model,image2,modelInfo)
        
        ####################################################################################
        
        endTime = time.time()
        print("Done, processing time:", "%.3f s"%(endTime-startTime))
        
        ############### Send image #####################################
        segImage  = segImage.astype(np.uint8)
        flatImage = segImage.flatten()
        hexaImage = bytes(flatImage)
        dataLen   = len(hexaImage)
        dataHex   = dataLen.to_bytes( 4, byteorder='big' )
        
        maskCenters = np.array(maskCenters).astype(np.uint32)
        flatCenters = maskCenters.flatten()
        byteCenters = bytes(flatCenters)
        centersLen  = len(byteCenters) 
        centersHex  = centersLen.to_bytes(4, byteorder='big')
        print("# cells:", "%i"%(len(maskCenters)))
        
        #signalState = 0
        empty_socket(conn)
        conn.send(b"SYN")
        message = conn.recv(7)
            
        if message.decode('utf-8') == "SYN+ACK":
            print("Transmiting data...")
            conn.send(dataHex)
            conn.send(hexaImage)
            message = conn.recv(3)
            if message.decode('utf-8') == "ACK":
                conn.send(centersHex)
                conn.send(byteCenters)
                signalState=1
                print("Done.")
                    
    except UnicodeDecodeError:
        print ('Encoding Error: restarting server')
    except TimeoutError:
        print ('Timeout Error: restarting server')
    except ConnectionResetError:
        print ('Connection Error: restarting server')
    except ConnectionAbortedError:
        print ('Client disconnected Error: restarting server')
    except:
        print ('Time out Error: restarting server')
    
    finally:
        conn.close()
        print('CONECTION CLOSED')
        print()
        