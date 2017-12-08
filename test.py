# Python 3
import jpeg as JPEG
import libs.ruidos as ruido
from skimage import io
import matplotlib.pyplot as plt
import seaborn as sns
# from util import *


img = io.imread('bmp/lena.bmp')

print(JPEG.img_size(img))
jpeg = JPEG.jpeg_encode(img,50)
print(jpeg.size())
# Decodifico
import pdb; pdb.set_trace()
img2 = JPEG.jpeg_decode(jpeg)
