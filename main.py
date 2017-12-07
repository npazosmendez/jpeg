# Python 3
from jpeg import *
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
from util import *

img = io.imread('bmp/lena.bmp')
print("Dimensiones originales:",img.shape)

# QTable = [[15,50],[70,200]]
QTable = np.array([\
    [16,11,10,16,24,40,51,61],\
    [12,12,14,19,26,58,60,55],\
    [14,13,16,24,40,57,69,56],\
    [14,17,22,29,51,87,80,62],\
    [18,22,37,56,68,109,103,77],\
    [24,35,55,64,81,104,113,92],\
    [49,64,78,87,103,121,120,101],\
    [72,92,95,98,112,100,103,99],\
    ])[0:4,0:4]
# Codifico en JPEG
jpeg = jpeg_encode(img,100)
# Decodifico
img2 = jpeg_decode(jpeg)

print("Dimensiones finales:",img2.shape)

plt.figure()
plt.imshow(img,cmap='gray')
plt.title('Sin comprimir\n size: '+str(len(img)*len(img[0])*3/1000)+' KB')

plt.figure()
plt.imshow(img2,cmap='gray')
plt.title('Comprimida con JPEG\n size: '+str(jpeg.size())+' KB')

plt.figure()
plt.imshow(img2-img)
plt.title('Diferencia')
plt.colorbar()

plt.show()
