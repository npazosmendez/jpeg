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


def block_qdct(img, NM, QTable):
    """
    ('block_qdct' = block quantized discrete cosine transformation)
    Aplica la transformada del coseno discreta a la imagen a cada cuadrado
    de NxM de la imagen de entrada, quantizando los valores según QTable.

    * Output: matriz de matrices de NxM. La unión de todas las matrices de NxM
    tiene el mismo tamaño que img.
    """

    N = NM[0]
    M = NM[1]

    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img - unos*128

    # Particiono en bloques
    print('Particionando en bloques',N,'x',M,'...')
    img_blocks = np.array_split(img,img.shape[0]//N,axis=0) # horizontal
    for i in range(len(img_blocks)):
        img_blocks[i] = np.array_split(img_blocks[i],img.shape[1]//M,axis=1) # vertical

    # Aplico DCT
    print('Calculando DCT...')
    img_blocks_qdct = np.copy(img_blocks)
    for i in range(len(img_blocks)):
        for j in range(len(img_blocks[0])):
            img_blocks_qdct[i][j] = dct2(img_blocks[i][j])

    # Cuantizo con la tabla
    print('Cuantizando coeficientes...')
    img_blocks_dctq = np.empty(img_blocks_qdct.shape, dtype = np.int8)
    for i in range(len(img_blocks)):
        for j in range(len(img_blocks[0])):
            img_blocks_dctq[i][j] = np.divide(img_blocks_qdct[i][j], QTable)
    img_blocks_dctq = np.array(img_blocks_dctq)

    return img_blocks_dctq

def block_qidct(img_blocks_dctq, QTable):
    """
    ('block_qidct' = block quantized inverse discrete cosine transformation)
    Aplica la anti transformada del coseno discreta a cada matriz de la
    matriz de matrices "img_blocks_dctq" y rearma la imagen original.

    * Output: imagen original, formada por las inversas del coseno de las
    matrices del input.
    """

    # Descuantizo con la tabla
    img_blocks_dct = np.zeros(img_blocks_dctq.shape, dtype = np.float)
    print('Descuantizando coeficientes...')
    k = 0
    for i in range(len(img_blocks_dct)):
        for j in range(len(img_blocks_dct[0])):
            img_blocks_dct[i][j] = np.multiply(img_blocks_dctq[i][j], QTable)
            k +=1

    # Aplico DCT
    print('Calculando IDCT...')
    img_blocks = np.zeros(img_blocks_dct.shape)
    for i in range(len(img_blocks)):
        for j in range(len(img_blocks[0])):
            img_blocks[i][j] = idct2(img_blocks_dct[i][j])

    # Uno los bloques
    img_blocks = np.concatenate(img_blocks.tolist(),axis=1)
    img = np.concatenate(img_blocks,axis=1)

    # Shifteo rango
    print('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int8)
    img = img + unos*128

    return img

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
    alto = img.shape[0]
    ancho = img.shape[1]

    # 1-padding para que altura%N=0 y ancho%M=0
    if not(img.shape[0] % N == 0):
        unos = np.ones((N-img.shape[0]%N,img.shape[1]),dtype=np.int8)
        img = np.concatenate((img, unos), axis=0)
    if not(img.shape[1] % M == 0):
        unos = np.ones((img.shape[0],M-img.shape[1]%M),dtype=np.int8)
        img = np.concatenate((img, unos), axis=1)

    # Cambio de espacio por bloque, cuantizado
    img_blocks_qdct = block_qdct(img,NM,QTable)

    # Codificación de coeficientes DC
    img_blocks_qdct = np.concatenate(img_blocks_qdct)
    print('Codificando coeficientes DC...')
    for i in range(len(img_blocks_qdct)-1,0,-1):
        img_blocks_qdct[i][0][0] = img_blocks_qdct[i][0][0] - img_blocks_qdct[i-1][0][0]

    # Obtención de secuencia zig-zag
    print('Obteniendo secuencia completa...')
    # TODO: por ahora está secuencial
    seq = np.concatenate(np.concatenate(img_blocks_qdct))

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
    img_blocks_dctq = np.array_split(seq,len(seq)//(N*M))
    for i in range(len(img_blocks_dctq)):
        img_blocks_dctq[i] = np.array_split(img_blocks_dctq[i],len(img_blocks_dctq[i])//N)
    img_blocks_dctq = np.array(img_blocks_dctq)

    # Decodificación de coeficientes DC
    print('Decodificando coeficientes DC...')
    for i in range(1,len(img_blocks_dctq)):
        img_blocks_dctq[i][0][0] = img_blocks_dctq[i-1][0][0] + img_blocks_dctq[i][0][0]

    # Armo la matrz de matrices y vuelvo a al imagen original
    img_blocks_dctq = np.array(np.array_split(img_blocks_dctq,alto//N))
    img = block_qidct(img_blocks_dctq,QTable)

    # Si hubo padding, se lo quito
    img = img[0:jpeg.height,0:jpeg.width]

    print('Decodificación finalizada.')
    return img
