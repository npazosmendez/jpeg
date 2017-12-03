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

def jpeg_encode(img, N = 8, M = None, QTable = None):
    """
    Codifica una imagen en mapa de bits a JPEG.
    * Input:
        img: array bidimensional (img de un solo canal, por ahora)
        N : natural (alto de bloque de partición, 8 por defecto)
        M : natural (ancho de bloque de partición, cuadrado por defecto)
        QTable: array bidimensional de NxM (tabla de cuantización de coeficientes)
    * Output:
        por verse...
    """
    # Establezo los valores por defecto
    if M == None:
        M = N
    if QTable == None:
        if not(N == 8 and M == 8):
            print('Falta tabla de cuantización.')
            exit(1)
        QTable = np.array([\
                [16,11,10,16,24,40,51,61],\
                [12,12,14,19,26,58,60,55],\
                [14,13,16,24,40,57,69,56],\
                [14,17,22,29,51,87,80,62],\
                [18,22,37,56,68,109,103,77],\
                [24,35,55,64,81,104,113,92],\
                [49,64,78,87,103,121,120,101],\
                [72,92,95,98,112,100,103,99],\
                ])
    assert(len(QTable) == N)
    assert(len(Qtable[0] == M))
    # Particiono en bloques
    blocks = block_partition(img, N, M)

    # Aplico DFT
    for i in range(len(blocks)):
        blocks[i] = fftpack.dct(blocks[i])

    # Cuantizo con la tabla
    for block in blocks:
        for u in range(len(block)):
            for v in range(len(block[0])):
                block[u][v] = int(block[u][v] / QTable[u][v])

    # Codificación de coeficientes DC
    for i in range(1,len(blocks)):
        blocks[i][0][0] = blocks[i-1][0][0]

    # Obtención de secuencia zig-zag
    # TODO

    # Codificación entrópica
    # TODO

img = io.imread('lena.bmp')[:,:,0]
img = block_partition(img, img.shape[0]//2, 100)
plt.imshow(img[0])
plt.show()
