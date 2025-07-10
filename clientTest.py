import socket
import numpy as np
import matplotlib.pyplot as plt
import select
import time
import struct

############### Load image ###############
im="temp/ImageFileName2.jpg"
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
hexaImage = ''.join([chr (q) for q in flatImage])
dataLen   = len(hexaImage.encode(encoding='utf-8'))
dataHex   = dataLen.to_bytes( 4, byteorder='big' )
################################################

# Creating a socket instance
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connecting to the localhost
ip_address = '127.0.0.1'
port = 5555


############ Connect #################
conn.connect((ip_address, port))

############ reicive SYN #############
message = conn.recv(7)
if message.decode('utf-8') == "SYN":
    ################# send SYN+AKC ###########
    conn.send(b"SYN+ACK")
    ################# send size ###########
    conn.send(ySize.to_bytes( 4, byteorder='big' ))
    conn.send(xSize.to_bytes( 4, byteorder='big' ))
    ################# send image ###########
    conn.send(hexaImage.encode(encoding='utf-8'))
    
    ################# receive AKC #############
    message = conn.recv(7)
    if message.decode('utf-8') == "ACK":
        print('Image send...')
        
        # receive image 
        ############### wait for SYN+ACK #################
        message = conn.recv(7)
        if message.decode('utf-8') == "SYN":
            conn.send(b"SYN+ACK")
            print("Receiving data...")
            dataHex = int.from_bytes(conn.recv(4), 'big')
            data_receive = conn.recv(dataHex)
            
            conn.send(b"ACK")
            
            centersHex  = int.from_bytes(conn.recv(4), 'big')
            byteCenters = conn.recv(centersHex)
            print("Done.")
            ######################################################## 
            centersInt = []
            if centersHex>0:
                
                for q in range(int(centersHex/4)):
                    centersInt.append(struct.unpack('<I', (byteCenters[q*4:(q*4)+4]))[0])
        
                centersArray = np.array(centersInt).reshape([int(centersHex/(4*3)),3])
    
                plt.figure()
                plt.imshow(image)
                plt.plot(centersArray[:,0],centersArray[:,1],'ro',)
                plt.show()
            
            
  