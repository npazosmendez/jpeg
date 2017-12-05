# Python 3
"""
No sé como organizar los fuentes por ahora porque no me imagino todo el código,
así que empiezo en este 'main.py'. Después se organizará todo mejor.
"""

"""
Los pasos a seguir en la codificación son:
* Para cada bloque de n x m (el estándar es 8x8):
    * Shiftear rango a -128 - 127
    * Aplicar FDCT a los n x m pixels
    * Cuantización de coeficientes DCT
    * Codificación de coeficientes DC (depende de los demás bloques)
    * Obtención de secuencia zig-zag y compresión
    * Codificación entrópica
"""

import numpy as np
from skimage import io
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from libs.huffman import *


def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

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
    res = np.array([img[x:x+M,y:y+N] for x in range(0,img.shape[0]-M+1,M) for y in range(0,img.shape[1]-N+1,N)])
    return res

def merge_blocks(blocks, alto, ancho, N = 8, M = None):
    if M == None:
        # Por defecto es cuadrado
        M = N
    res = []
    for x in range(0,len(blocks),ancho//M):
        res.append(np.concatenate(blocks[x:x+ancho//M],axis=1))
    res = np.array(res)
    res = np.concatenate(res,axis=0)
    return res

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

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
    assert(len(QTable[0]) == M)


    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img - unos*128

    # Particiono en bloques
    print('Particionando en bloques',N,'x',M,'...')
    blocks = block_partition(img, N, M)

    # Aplico DCT
    print('Calculando DCT...')
    blocks_dct = np.zeros(blocks.shape)
    for i in range(len(blocks)):
        blocks_dct[i] = dct2(blocks[i])

    # Cuantizo con la tabla
    print('Cuantizando coeficientes...')
    blocks_dctq = np.empty(blocks_dct.shape, dtype = np.int8)
    for i in range(len(blocks_dct)):
        blocks_dctq[i] = np.divide(blocks_dct[i], QTable)

    # Codificación de coeficientes DC
    print('Codificando coeficientes DC...')
    for i in range(len(blocks_dctq)-1,0,-1):
        blocks_dctq[i][0][0] = blocks_dctq[i][0][0] - blocks_dctq[i-1][0][0]

    # Obtención de secuencia zig-zag
    print('Obteniendo secuencia completa...')
    # TODO: por ahora está secuencial
    seq = np.concatenate(np.concatenate(blocks_dctq))

    # Compresión de secuencia
    # cada simbolo K se reduce a una tupla (C,k), donde C indica la cantidad
    # de ceros que tiene antes (por bloque)
    seq_comp = []
    i = 0
    while i < len(seq):
        ceros = 0
        j = 0
        while j < N*M:
            if seq[i] == 0:
                ceros+=1
            else:
                tupla = (ceros, seq[i])
                seq_comp.append(tupla)
                ceros = 0
            i+=1
            j+=1
        seq_comp.append((0,0))

    # Codificación entrópica
    print('Comprimiendo...')
    (binstring, hufftree) = huffman_compress(seq_comp)

    print('Compresión finalizada.')
    print('Tamaño en Kbytes original: \t', img.shape[0]*img.shape[1]*8 / 8 / 1000)
    print('Tamaño en Kbytes comprimido: \t', len(binstring) / 8 / 1000)

    return (binstring, hufftree)


def jpeg_decode(binstring, hufftree, alto, ancho, N = 8, M = None, QTable = None):
    """
    Decodifica una imagen de JPEG a mapa de bits.
    * Input (sujeto a modificaciones):
        binstring: string de 0/1 (datos de la imagen codificados)
        hufftree: árbol de codificación de Huffman
        N : natural (alto de bloque de partición, 8 por defecto)
        M : natural (ancho de bloque de partición, cuadrado por defecto)
        QTable: array bidimensional de NxM (tabla de cuantización de coeficientes)
    * Output: imagen en mapa de bits
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
    assert(len(QTable[0]) == M)

    # Decodificación entrópica
    print('Descomprimiendo...')
    seq_comp = huffman_uncompress(binstring, hufftree)

    # Descompresión de secuencia
    # cada simbolo K se reduce a una tupla (C,k), donde C indica la cantidad
    # de ceros que tiene antes (por bloque)
    seq = []
    cuantos = 0
    for (C,k) in seq_comp:
        if (C,k) == (0,0):
            for i in range(N*M-cuantos):
                seq.append(0)
            cuantos = 0
        else:
            for i in range(C):
                seq.append(0)
            seq.append(k)
            cuantos += C + 1

    # Obtención de secuencia zig-zag
    print('Desarmando secuencia...')
    # TODO: por ahora está secuencial

    seq = np.array(seq)
    blocks_dctq = np.split(seq,len(seq)//(N*M))
    for i in range(len(blocks_dctq)):
        blocks_dctq[i] = np.split(blocks_dctq[i],len(blocks_dctq[i])//N)
    blocks_dctq = np.array(blocks_dctq)

    # Decodificación de coeficientes DC
    print('Decodificando coeficientes DC...')
    for i in range(1,len(blocks_dctq)):
        blocks_dctq[i][0][0] = blocks_dctq[i-1][0][0] + blocks_dctq[i][0][0]

    # Cuantizo con la tabla
    blocks_dct = np.zeros(blocks_dctq.shape, dtype = np.float)
    print('Descuantizando coeficientes...')
    for i in range(len(blocks_dct)):
        blocks_dct[i] = np.multiply(blocks_dctq[i], QTable)

    # Aplico DCT
    print('Calculando IDCT...')
    blocks = np.zeros(blocks_dct.shape)
    for i in range(len(blocks)):
        blocks[i] = idct2(blocks_dct[i])

    # Uno los bloques
    print('Uniendo los bloques de ',N,'x',M,'...')
    img = merge_blocks(blocks,alto,ancho,N,M)

    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img + unos*128

    print('Decodificación finalizada.')
    return img


############################################################
############################################################
############################################################


img = io.imread('bmp/lena.bmp')[:,:,0]

plt.figure()
plt.imshow(img,cmap='gray')
plt.title('Sin comprimir\n size: '+str(len(img)*len(img[0])/1000)+' KB')


QTable = [[50,50],[70,50]]
# Codifico en JPEG
(binstring, hufftree) = jpeg_encode(img,N=2,M=2,QTable=QTable)
# Decodifico
img = jpeg_decode(binstring, hufftree,len(img),len(img[0]),N=2,M=2,QTable=QTable)


plt.figure()
plt.imshow(img,cmap='gray')
plt.title('Comprimida con JPEG\n size: '+str(len(binstring)/8/1000)+' KB')

plt.show()
