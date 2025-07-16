import socket
import numpy as np
import matplotlib.pyplot as plt
import select
import time
import struct
import json

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionResetError("Socket closed before receiving full data")
        data += more
    return data

############### Load image ###############
im="temp/ImageFileName5.jpg"
original = plt.imread(im)

ySize,xSize=np.shape(original)
image = original

plt.figure()
plt.imshow(image)
plt.show()
###########################################

############# Image in bytes ###################
image     = image.astype(np.uint8)
flatImage = image.flatten()
hexaImage = bytes(flatImage)
dataLen   = len(hexaImage)
dataHex   = dataLen.to_bytes( 4, byteorder='big' )
################################################

########### Model information #################
modelInfo = {
    "model"     : "Yolo5_01",
    "confThrd"  : 0.51      ,
    "minArea"   : 5000      ,
    "minBorDist": 50        ,
    "maxOverlap": 0.9       ,
    "maxIoU"    : 0.35      ,
    "resolution": 120
}
json_object = json.dumps(modelInfo, indent=4)
infoBytes   = json_object.encode('ascii')
infoLen     = len(infoBytes)
###############################################

# Connecting to the localhost
ip_address = '10.3.20.25'
#ip_address = '10.2.9.28'
#ip_address = 'isrvlusersrv01'
port = 5555

print(socket.getaddrinfo(ip_address, port))

startTime  = time.time()
sleepTime  = 1
centersHex = 0
for _ in range(10):
    try:
        ##### Creating a socket instance #####
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ############ Connect #################
        conn.connect((ip_address, port))
        conn.settimeout(10)
        ############ reicive SYN #############
        message = conn.recv(7)
        if message.decode('utf-8') == "SYN":
            ############### send SYN+AKC ###########
            conn.send(b"SYN+ACK")
            ############### send size    ###########
            conn.send(ySize.to_bytes( 4, byteorder='big' ))
            conn.send(xSize.to_bytes( 4, byteorder='big' ))
            ################# send image ###########
            conn.send(hexaImage)
            ################# receive AKC #############
            message = conn.recv(7)
            if message.decode('utf-8') == "ACK":
                print('Image send...')
                
                ######### send Information ###########
                ############ reicive SYN #############
                message = conn.recv(7)
                if message.decode('utf-8') == "SYN":
                    ############### send SYN+AKC ###########
                    conn.send(b"SYN+ACK")
                    ############### send size    ###########
                    conn.send(infoLen.to_bytes( 4, byteorder='big' ))
                    ################# send image ###########
                    conn.send(infoBytes)
                    ################# receive AKC #############
                    message = conn.recv(7)
                    if message.decode('utf-8') == "ACK":
                        print('Model Information send...')
                
                # receive image 
                ############### wait for SYN+ACK #################
                message = conn.recv(7)
                if message.decode('utf-8') == "SYN":
                    conn.send(b"SYN+ACK")
                    print("Receiving data...")
                    dataHex = int.from_bytes(conn.recv(4), 'big')
                    #data_receive = conn.recv(dataHex)
                    data_receive = recv_all(conn, dataHex)
                    #print(len(data_receive))
                    
                    conn.send(b"ACK")
                    
                    centersHex  = int.from_bytes(conn.recv(4), 'big')
                    #byteCenters = conn.recv(centersHex)
                    byteCenters = recv_all(conn, centersHex)
                
                    if centersHex==len(byteCenters) and centersHex!=0:
                        print("Done.")
                        break
                    else:
                        raise ConnectionResetError
                        print("Corrupt data, restarting ...")
    except ConnectionResetError:
        print ('Connection Error: trying again')
    except OSError:
        conn.close()
        time.sleep(sleepTime)
        print ('Server busy: trying again')
        sleepTime += 0.5
######################################################################### 

endTime = time.time()
print("Processing time:", "%.3f s"%(endTime-startTime))
              
centersInt = []
for q in range(int(centersHex/4)):
    centersInt.append(struct.unpack('<I', (byteCenters[q*4:(q*4)+4]))[0])
centersArray = np.array(centersInt).reshape([int(centersHex/(4*3)),3])   
print(centersArray)  
  
plt.figure()
plt.imshow(image)
if centersInt != [0, 0, 0]:   
    plt.plot(centersArray[:,0],centersArray[:,1],'ro',)
plt.show()
            

data_receive 
            
  