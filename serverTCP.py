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

def processImage(model,image2, cellsBorder=True):
    original = image2 
    results  = model(image2)
    #results.show()

    masks_np = results.pandas().xyxy[0].sort_values("ymin")

    confThrd = 0.51
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
        if row.confidence > confThrd and area > minArea:
            if cellsBorder:
                if xMean > minBorDist and yMean > minBorDist:
                    if xMean < (original.shape[1]-minBorDist):
                        if yMean < (original.shape[0]-minBorDist):
                            centers.append([xMean,yMean,row.xmin])
                            segment[np.where(((xv>row.xmin)*1 + (xv<row.xmax)*1 + (yv>row.ymin)*1 + (yv<row.ymax)*1)==4)]=(count%255)
                            count += 1
            else:
                if row.xmax < (original.shape[1]-minBorDist/2) and row.xmin > minBorDist/2:
                    if row.ymax < (original.shape[0]-minBorDist/2) and row.ymin > minBorDist/2:
                        centers.append([xMean,yMean,row.xmin])
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
    conn.settimeout(5)
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
            segImage,maskCenters  = processImage(model,image2)
        except:
            print('########## Model Error ################')
            segImage,maskCenters  = processImage(model,image2)
        
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
        
        signalState = 0
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
        