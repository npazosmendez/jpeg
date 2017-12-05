# Python 3
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
from scipy.fftpack import dct, idct
from libs.huffman import *

class jpeg:
    def __init__(self, binstring, hufftree, height, width, NM = (8,8), \
                QTable = np.array([\
                    [16,11,10,16,24,40,51,61],\
                    [12,12,14,19,26,58,60,55],\
                    [14,13,16,24,40,57,69,56],\
                    [14,17,22,29,51,87,80,62],\
                    [18,22,37,56,68,109,103,77],\
                    [24,35,55,64,81,104,113,92],\
                    [49,64,78,87,103,121,120,101],\
                    [72,92,95,98,112,100,103,99],\
                    ])):
        assert(len(QTable) == NM[0])
        assert(len(QTable[0]) == NM[1])
        self.binstring = binstring
        self.height = height
        self.width = width
        self.data = binstring
        self.hufftree = hufftree
        self.N = NM[0]
        self.M = NM[1]
        self.QTable = QTable

    def size(self):
        return len(self.binstring)/8/1000


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def jpeg_encode(img, NM = (8,8), QTable = np.array([\
        [16,11,10,16,24,40,51,61],\
        [12,12,14,19,26,58,60,55],\
        [14,13,16,24,40,57,69,56],\
        [14,17,22,29,51,87,80,62],\
        [18,22,37,56,68,109,103,77],\
        [24,35,55,64,81,104,113,92],\
        [49,64,78,87,103,121,120,101],\
        [72,92,95,98,112,100,103,99],\
        ])):
    """
    Codifica una imagen en mapa de bits a JPEG.
    * Input:
        img: array bidimensional (img de un solo canal, por ahora)
        NM : tupla de naturales (tamaño de bloque, (8,8) por defecto)
        QTable: array bidimensional de NxM (tabla de cuantización de coeficientes)
    * Output: instancia de la clase jpeg
    """
    assert(len(QTable) == NM[0])
    assert(len(QTable[0]) == NM[1])

    N = NM[0]
    M = NM[1]
    # 1-padding para que altura % N = 0 y ancho % M = 0
    alto = img.shape[0]
    ancho = img.shape[1]
    if not(img.shape[0] % N == 0):
        unos = np.ones((N-img.shape[0]%N,img.shape[1]),dtype=np.int8)
        img = np.concatenate((img, unos), axis=0)

    if not(img.shape[1] % M == 0):
        unos = np.ones((img.shape[0],M-img.shape[1]%M),dtype=np.int8)
        img = np.concatenate((img, unos), axis=1)

    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img - unos*128

    # Particiono en bloques
    print('Particionando en bloques',N,'x',M,'...')
    blocks = np.array_split(img,img.shape[0]//N,axis=0) # horizontal
    for i in range(len(blocks)):
        blocks[i] = np.array_split(blocks[i],img.shape[1]//M,axis=1) # vertical

    # Aplico DCT
    print('Calculando DCT...')
    blocks_dct = np.copy(blocks)
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            blocks_dct[i][j] = dct2(blocks[i][j])

    # Cuantizo con la tabla
    print('Cuantizando coeficientes...')
    blocks_dctq = np.empty(blocks_dct.shape, dtype = np.int8)
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            blocks_dctq[i][j] = np.divide(blocks_dct[i][j], QTable)


    # Codificación de coeficientes DC
    blocks_dctq = np.concatenate(blocks_dctq)
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

    res = jpeg(binstring, hufftree, alto, ancho, (N,M), QTable)

    return res

def jpeg_decode(jpeg):
    """
    Decodifica una imagen de JPEG a mapa de bits.
    * Input (sujeto a modificaciones):
        jpeg: instancia de la clase jpeg
    * Output: imagen en mapa de bits
    """
    binstring = jpeg.binstring
    hufftree = jpeg.hufftree
    alto = jpeg.height
    ancho = jpeg.width
    N = jpeg.N
    M = jpeg.M
    QTable = jpeg.QTable

    # Tomo alturas auxiliares, por si hay padding
    if not(alto % N == 0):
        alto = alto + N-alto%N
    if not(ancho % M == 0):
        ancho = ancho + M-ancho%M

    # Decodificación entrópica
    print('Descomprimiendo...')
    seq_comp = huffman_uncompress(binstring, hufftree)

    # Descompresión de secuencia
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
    blocks_dctq = np.array_split(seq,len(seq)//(N*M))
    for i in range(len(blocks_dctq)):
        blocks_dctq[i] = np.array_split(blocks_dctq[i],len(blocks_dctq[i])//N)
    blocks_dctq = np.array(blocks_dctq)

    # Decodificación de coeficientes DC
    print('Decodificando coeficientes DC...')
    for i in range(1,len(blocks_dctq)):
        blocks_dctq[i][0][0] = blocks_dctq[i-1][0][0] + blocks_dctq[i][0][0]

    # Cuantizo con la tabla
    blocks_dct = np.zeros((alto//N,ancho//M,N,M), dtype = np.float)
    print('Descuantizando coeficientes...')
    k = 0
    for i in range(len(blocks_dct)):
        for j in range(len(blocks_dct[0])):
            blocks_dct[i][j] = np.multiply(blocks_dctq[k], QTable)
            k +=1

    # Aplico DCT
    print('Calculando IDCT...')
    blocks = np.zeros(blocks_dct.shape)
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            blocks[i][j] = idct2(blocks_dct[i][j])

    # Uno los bloques
    print('Uniendo los bloques de ',N,'x',M,'...')
    blocks = np.concatenate(blocks.tolist(),axis=1)
    img = np.concatenate(blocks,axis=1)

    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img + unos*128

    # Si hubo padding, se lo quito
    img = img[0:jpeg.height,0:jpeg.width]

    print('Decodificación finalizada.')
    return img
