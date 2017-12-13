import numpy as np
from collections import Counter

# Información
def entropy(data):
    data = np.ravel(data)
    n = len(data)
    probs = np.array(list(Counter(data).values()))/n # Calculo los pesos
    entropy = - sum(probs * np.log2(probs))
    return entropy


# Error
def PSNR(img_original, img_aprox):
    return 10*np.log10(255.0**2/MSE(img_original, img_aprox))

def MSE(v_original, v_aprox):
    v_original = np.ravel(v_original)
    v_aprox = np.ravel(v_aprox)
    assert(len(v_original) == len(v_aprox))
    n = len(v_original)
    return 1/n*sum(np.multiply(v_original - v_aprox, v_original - v_aprox))

# Compresión
def FC(bytes_original, bytes_comp):
    return bytes_original/bytes_comp
