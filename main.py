# Python 3
from jpeg import *
from skimage import io
import matplotlib.pyplot as plt


import numpy as np
img = io.imread('bmp/sun.bmp')[:,:,0]


# QTable = np.array([\
#         [16,11,10,16,24,40,51,61],\
#         [12,12,14,19,26,58,60,55],\
#         [14,13,16,24,40,57,69,56],\
#         [14,17,22,29,51,87,80,62],\
#         [18,22,37,56,68,109,103,77],\
#         [24,35,55,64,81,104,113,92],\
#         [49,64,78,87,103,121,120,101],\
#         [72,92,95,98,112,100,103,99],\
#         ])
#
# asd = block_qdct(img, (8,8), QTable)
#
# img2 = block_qidct(asd, QTable)
#
# plt.subplot(2,1,1)
# plt.imshow(img)
# plt.subplot(2,1,2)
# plt.imshow(img2)
# plt.show()
#
# exit()


print("Dimensiones originales:",img.shape)

plt.figure()
plt.imshow(img,cmap='gray')
plt.title('Sin comprimir\n size: '+str(len(img)*len(img[0])/1000)+' KB')


# QTable = [[15,50],[70,200]]
# Codifico en JPEG
jpeg = jpeg_encode(img,(8,8))
# Decodifico
img = jpeg_decode(jpeg)

print("Dimensiones finales:",img.shape)

plt.figure()
plt.imshow(img,cmap='gray')
plt.title('Comprimida con JPEG\n size: '+str(jpeg.size())+' KB')

plt.show()
