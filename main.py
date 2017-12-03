# Python 3
"""
No sé como organizar los fuentes por ahora porque no me imagino todo el código,
así que empiezo en este 'main.py'. Después se organizará todo mejor.
"""

"""
Los pasos a seguir en la codificación son:
* Para cada bloque de n x m (el estándar es 8x8):
    * Aplicar FDCT a los n x m pixels
    * Cuantización de coeficientes DCT
    * Codificación de coeficientes DC (depende de los demás bloques)
    * Obtención de secuencia zig-zag
    * Codificación entrópica
"""

import numpy as np
from skimage import io
from scipy import fftpack
import matplotlib.pyplot as plt


def block_partition(img, N = 8, M = None):
    """
    Particiona la imagen en bloques de N x M
    * Input:
        img : array bidimensional (img de un solo canal)
        N : natural (alto de bloque, 8 por defecto)
        M : natural (ancho de bloque, cuadrado por defecto)

    * Output: array de '(Height/N * Width/N) x N x M' (array de matrices)
    """
    if M == None:
        # Por defecto es cuadrado
        M = N
    res = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
    return res


img = io.imread('lena.bmp')[:,:,0]
img = block_partition(img, img.shape[0]//2, 100)
plt.imshow(img[0])
plt.show()
