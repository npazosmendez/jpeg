"""
Biblioteca de generadoras de ruido
"""

import numpy as np
import random

def add_gaussian_noise(img, mean, sd):
    IMG = np.copy(img)
    for i in range(len(IMG)):
        for j in range(len(IMG[0])):
            IMG[i,j] += int(random.gauss(mean, sd))
    return IMG

def mult_rayleigh_noise(img, mean, var, dist_vals = None):
    b = (4 * var)/(4-np.pi)
    a = mean - np.sqrt(np.pi*b/4)
    print("a = %f , b = %f" % (a, b))
    IMG = np.copy(img)
    dist = lambda u: a + np.sqrt(-b*np.log(1-u))
    for i in range(len(IMG)):
        for j in range(len(IMG[0])):
            val = dist(random.uniform(0,1))
            if dist_vals is not None: dist_vals.append(val)
            IMG[i,j] = int(float(IMG[i,j]) * val)
    return IMG

def mult_rayleigh_noise(img, xi):
    IMG = np.copy(img)
    dist = lambda u: xi*np.sqrt(-2*np.log(u))
    for i in range(len(IMG)):
        for j in range(len(IMG[0])):
            val = dist(random.uniform(0,1))
            IMG[i,j] = int(float(IMG[i,j]) * val)
    return IMG

# P&S noise tambien es conocido como ruido impulsivo
def pep_salt_noise(img, prob):
    IMG = np.copy(img)
    p_p = prob/2; p_s=1-prob/2
    for i in range(len(IMG)):
        for j in range(len(IMG[0])):
            lottery = np.random.uniform()
            if lottery < p_p:
                # pepper
                IMG[i,j] = 0
            elif p_s < lottery:
                #salt
                IMG[i,j] = 255                
            else:
                #none
                continue    
    return IMG
