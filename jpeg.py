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
import huffman
from libs.color import *
from collections import Counter
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

DEBUG = False

def dprint(msg, *demases):
    if DEBUG:
        print(msg,demases)

def img_size(img):
    # cant_canales * alto * ancho / 1000 -> [peso en kB]
    if len(img.shape) > 2:
        # la imagen tiene mas de un canal
        return img.shape[2]*img.shape[0]*img.shape[1] / 1000
    else:
        # la imagen es monocroma
        return img.shape[0]*img.shape[1] / 1000


class jpeg:
    def __init__(self, YCbCr_binstring, YCbCr_huffdic, height, width, NM = (8,8), \
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
        self.Ybinstring = YCbCr_binstring[0]
        self.Cbbinstring = YCbCr_binstring[1]
        self.Crbinstring = YCbCr_binstring[2]
        self.Yhuffdic = YCbCr_huffdic[0]
        self.Cbhuffdic = YCbCr_huffdic[1]
        self.Crhuffdic = YCbCr_huffdic[2]
        self.height = height
        self.width = width
        self.N = NM[0]
        self.M = NM[1]
        self.QTable = QTable

    def size(self):
        return (len(self.Ybinstring) + len(self.Cbbinstring) + len(self.Crbinstring))/ 8 /1000


def jpeg_encode(img, Q = 100, NM = (8,8), QTable = np.array([\
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
    Comprime y codifica la imagen de 3 canales "img" de mapa de bits a JPEG.
    * Input:
        img: matriz de HxWx3 (la imagen de 3 canales a codificar)
        Q: entero de 1 a 100 (calidad de imagen comprimida)
        NM: tupla de enteros (tamaño de bloque del algoritmo JPEG)
        QTable: matriz de NxM (tabla de cuantización de coeficientes)
    * Output: instancia de clase jpeg.
    """
    assert(0 < Q and 100 >= Q)
    factor = (100 - Q + 1)/2

    if not(len(QTable)==NM[0]) or not(len(QTable[0])==NM[1]):
        dprint(' !! Adaptando tabla de cuantización...')
        QTable = zoom(QTable,[NM[0]/len(QTable),NM[1]/len(QTable[0])])

    # Convierto a espacio YCbCr
    img_YCbCr = RGB2YCbCr(img)
    img_Y = img_YCbCr[:,:,0]
    img_Cb = img_YCbCr[:,:,1]
    img_Cr = img_YCbCr[:,:,2]

    # Codifico los tres canales como JPEG en tira de bits
    (Ybinstring, Yhuffdic) = jpeg_mono_encode(img_Y, NM, QTable*factor)
    (Cbbinstring, Cbhuffdic) = jpeg_mono_encode(img_Cb, NM, QTable*factor)
    (Crbinstring, Crhuffdic) = jpeg_mono_encode(img_Cr, NM, QTable*factor)

    # Creo la instancia JPEG con los tres canales
    height = img.shape[0]
    width = img.shape[1]
    YCbCr_binstring = (Ybinstring, Cbbinstring, Crbinstring)
    YCbCr_huffdic = (Yhuffdic, Cbhuffdic, Crhuffdic)

    res = jpeg(YCbCr_binstring, YCbCr_huffdic, height, width, NM, QTable*factor)

    return res

def jpeg_decode(jpeg):
    """
    Decodifica una instancia de la clase 'jpeg' a una imagen en mapa de bits.
    * Input:
        jpeg: instancia de clase 'jpeg'
    * Output: matriz de HxWx3 (los 3 canales de la imagen en mapa de bits)
    """

    # Decodifico de JPEG a mapa de bits cada canal Y, Cb, Cr
    img_Y = jpeg_mono_decode(jpeg.Ybinstring, jpeg.Yhuffdic, (jpeg.N, jpeg.M), jpeg.QTable, jpeg.height, jpeg.width)
    img_Cb = jpeg_mono_decode(jpeg.Cbbinstring, jpeg.Cbhuffdic, (jpeg.N, jpeg.M), jpeg.QTable, jpeg.height, jpeg.width)
    img_Cr = jpeg_mono_decode(jpeg.Crbinstring, jpeg.Crhuffdic, (jpeg.N, jpeg.M), jpeg.QTable, jpeg.height, jpeg.width)

    # Uno los tres canales en una matrix de alto x ancho x 3
    img_YCbCr = np.empty((img_Y.shape[0],img_Y.shape[1],3), dtype= np.uint8)
    img_YCbCr[:,:,0] = img_Y
    img_YCbCr[:,:,1] = img_Cb
    img_YCbCr[:,:,2] = img_Cr

    # Cambio de espacio de color: YCbCr -> RGB
    img_RGB = YCbCr2RGB(img_YCbCr)

    return img_RGB

def jpeg_mono_encode(img, NM, QTable):
    """
    Codifica una imagen de un único canal en mapa de bits a JPEG.
    Esta función no debería usarse directamente por el usuario. Para codificar
    a JPEG véase 'jpeg_encode'.
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
    dprint('Codificando coeficientes DC...')
    for i in range(len(img_blocks_qdct)-1,0,-1):
        img_blocks_qdct[i][0][0] = img_blocks_qdct[i][0][0] - img_blocks_qdct[i-1][0][0]

    # Obtención de secuencia zig-zag
    dprint('Obteniendo secuencia completa...')
    if True:
        # zig-zag
        seq = fast_ZZPACK(img_blocks_qdct)
    else:
        # secuencial
        seq = np.concatenate(np.concatenate(img_blocks_qdct))

    # Codificación entrópica
    dprint('Comprimiendo...')
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

    (binstring, huffdic) = huffman_compress(seq_comp)

    dprint('Compresión finalizada.')
    dprint('Tamaño en Kbytes original: \t', img_size(img))
    dprint('Tamaño en Kbytes comprimido: \t', len(binstring) / 8 / 1000)

    res = (binstring, huffdic)
    return res

def jpeg_mono_decode(binstring, huffdic, NM, QTable, height, width):
    """
    Decodifica una imagen de JPEG a mapa de bits.
    Esta función no debería usarse directamente por el usuario. Para decodificar
    JPEG véase 'jpeg_decode'.
    * Input:
        binstring: string de 0/1 (datos que representan la imagen en jpeg)
        huffdic: diccionario string de 0/1 -> símbolo (para decodificar binstring)
        NM: tupla de naturales
        QTable: matriz de NxM
        height: natural (altura de la imagen)
        width: natural (ancho de la imagen)
    * Output: imagen en mapa de bits
    """
    height_aux = height
    width_aux = width
    N = NM[0]
    M = NM[1]

    # Tomo alturas auxiliares, por si hay padding
    if not(height_aux % N == 0):
        height_aux = height_aux + N-height_aux%N
    if not(width_aux % M == 0):
        width_aux = width_aux + M-width_aux%M

    # Decodificación entrópica
    dprint('Descomprimiendo...')
    seq_comp = huffman_uncompress(binstring, huffdic)

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

    # Obtención de matrices a partir de secuencias zig-zag
    dprint('Desarmando secuencia...')
    if True:
        # zig-zag
        img_blocks_dctq = zig_zag_unpacking(seq,N,M)
    else:
        # secuencial
        seq = np.array(seq)
        img_blocks_dctq = np.array_split(seq,len(seq)//(N*M))
        for i in range(len(img_blocks_dctq)):
            img_blocks_dctq[i] = np.array_split(img_blocks_dctq[i],len(img_blocks_dctq[i])//N)
        img_blocks_dctq = np.array(img_blocks_dctq)


    # Decodificación de coeficientes DC
    dprint('Decodificando coeficientes DC...')
    for i in range(1,len(img_blocks_dctq)):
        # aca img_blocks_dctq ses de dype=np.uint8
        # para mi aca hay algo medio raro con esto:
            # https://stackoverflow.com/questions/7559595/python-runtimewarning-overflow-encountered-in-long-scalars
        # dice que en python hicieron su propio check de overflow
        limit_dac = 255 - img_blocks_dctq[i-1][0][0]
        if  img_blocks_dctq[i][0][0] > limit_dac:
            img_blocks_dctq[i][0][0] = 255
        else:
            img_blocks_dctq[i][0][0] = img_blocks_dctq[i-1][0][0] + img_blocks_dctq[i][0][0]

    # Armo la matrz de matrices y vuelvo a al imagen original
    img_blocks_dctq = np.array(np.array_split(img_blocks_dctq,height_aux//N))
    img = block_qidct(img_blocks_dctq,QTable)

    # Si hubo padding, se lo quito
    img = img[0:height,0:width]

    dprint('Decodificación finalizada.')
    return img


def fast_ZZPACK(blocks):
    """
    Toma un arreglo de matrices y devuelve un arreglo de números que surge de
    recorrer en forma de zig-zag las matrices del arreglo en orden.

    Input:
        * blocks: arreglo de matrices, todas de igual tamaño
    Output: arreglo de enteros
    """
    OUT = np.concatenate([np.diagonal(block[::-1,:], k)[::(2*(k % 2)-1)] for block in blocks for k in range(1-block.shape[0], block.shape[0])])
    return OUT


def zig_zag_unpacking(zig_zagged_array,N,M):
    """
    Toma una secuencia 'zig_zagged_array' y devuelve una lista de matrices de NxM, que se
    arman de interpretar subsecuencias de NxM elementos de 'zig_zagged_array' como
    recorridos zig-zag de varias matrices.
    Input:
    * zig_zagged_array: arreglo de enteros
    * N,M : naturales
    Precondición: largo(zig_zagged_array) es múltiplo de N*M
    Output: arreglo de matrices de NxM
    """

    # Variable a devolver
    block_array = np.empty((len(zig_zagged_array)//(N*M),N,M),dtype=np.int8)

    # Creo el índice para reordenar los elementos
    MAT = np.array(np.split(np.arange(0,N*M),M)) # matriz ordenada
    ZIG_MAT_RAV = np.concatenate([np.diagonal(MAT[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-MAT.shape[0], MAT.shape[0])]) # matriz recorrida en diagonal aplanada

    # Reordeno de a NxM elementos según el índice 'ZIG_MAT_RAV' y los guardo en 'block'.
    block = np.zeros(N*M,dtype=np.int)
    for k in range(len(zig_zagged_array)//(N*M)): # qty of blocks in array
        for i in range(N*M):
            block[ZIG_MAT_RAV[i]] = zig_zagged_array[k*N*M+i]
        # Luego pongo 'block' en el resultado
        block_array[k] = np.array(np.split(block,M))

    # Convierto salida en np.array
    block_array = np.array(block_array)

    return block_array

def huffman_compress(seq):
    probs = list(Counter(seq).items()) # Calculo los pesos
    # Si hay un solo simbolo, huffman considera que no hay
    # información para codificar. Caso aparte:
    if len(probs) == 1:
        binstring = '1'*probs[0][1]
        huffdic_decode = {'1':probs[0][0]}
        return (binstring,huffdic_decode)
    huffdic_encode = huffman.codebook(probs)
    huffdic_decode = {}
    binstring = ""
    for x in seq:
        binstring = binstring + huffdic_encode[x]
        huffdic_decode[huffdic_encode[x]] = x
    return (binstring,huffdic_decode)

def huffman_uncompress(binstring, huffdic):
    seq = []
    string = ""
    for i in range(len(binstring)):
        string = string + binstring[i]
        if string in huffdic:
            seq.append(huffdic[string])
            string = ""
    return seq


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
    dprint('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int)
    img = img - unos*128

    # Particiono en bloques
    dprint('Particionando en bloques',N,'x',M,'...')
    img_blocks = np.array_split(img,img.shape[0]//N,axis=0) # horizontal
    for i in range(len(img_blocks)):
        img_blocks[i] = np.array_split(img_blocks[i],img.shape[1]//M,axis=1) # vertical

    # Aplico DCT
    dprint('Calculando DCT...')
    img_blocks_qdct = np.copy(img_blocks)
    for i in range(len(img_blocks)):
        for j in range(len(img_blocks[0])):
            img_blocks_qdct[i][j] = dct2(img_blocks[i][j])

    # Cuantizo con la tabla
    dprint('Cuantizando coeficientes...')
    img_blocks_dctq = np.empty(img_blocks_qdct.shape, dtype = np.int)
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
    img_blocks_shape = (img_blocks_dctq.shape[0], img_blocks_dctq.shape[1])
    img_blocks_dct = np.empty(img_blocks_shape, dtype = np.ndarray)
    dprint('Descuantizando coeficientes...')
    for i in range(len(img_blocks_dct)):
        for j in range(len(img_blocks_dct[0])):
            img_blocks_dct[i][j] = np.multiply(img_blocks_dctq[i][j], QTable)

    # Aplico DCT
    dprint('Calculando IDCT...')
    img_blocks = np.empty(img_blocks_dct.shape,dtype=np.ndarray)
    for i in range(len(img_blocks)):
        for j in range(len(img_blocks[0])):
            img_blocks[i][j] = np.clip(idct2(img_blocks_dct[i][j]),-128,127)

    # Uno los bloques
    img_blocks_conc = np.concatenate(img_blocks.tolist(),axis=1)
    img = np.concatenate(img_blocks_conc,axis=1)

    # Shifteo rango
    dprint('Shifteando rangos...')
    unos = np.ones(img.shape,dtype=np.int)
    img = img + unos*128

    return img
