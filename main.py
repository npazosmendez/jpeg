# Python 3
from jpeg import *
from skimage import io
import matplotlib.pyplot as plt


import numpy as np
img = io.imread('bmp/sun.bmp')[:,:,0]

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
