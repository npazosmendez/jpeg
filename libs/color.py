import numpy as np

def RGB2YCbCr(img):
    # trabajo en ints para no sufrir overflow
    R = np.array(img[:,:,0], dtype=np.int)
    G = np.array(img[:,:,1], dtype=np.int)
    B = np.array(img[:,:,2], dtype=np.int)

    Y = np.array(np.clip(0.299*R + 0.587*G + 0.114*B,0,255), dtype=np.uint8)
    Cb = np.array(np.clip(128 - 0.168736*R - 0.331264*G + 0.5*B,0,255), dtype=np.uint8)
    Cr = np.array(np.clip(128 + 0.5*R - 0.418688*G - 0.081312*B,0,255), dtype=np.uint8)

    img_YCbCr = np.empty((Y.shape[0],Y.shape[1],3), dtype= np.uint8)
    img_YCbCr[:,:,0] = Y
    img_YCbCr[:,:,1] = Cb
    img_YCbCr[:,:,2] = Cr
    return img_YCbCr

def YCbCr2RGB(img):
    # trabajo en ints para no sufrir overflow
    Y = np.array(img[:,:,0], dtype=np.int)
    Cb = np.array(img[:,:,1], dtype=np.int)
    Cr = np.array(img[:,:,2], dtype=np.int)

    R = np.array(Y + 1.402*(Cr-128), dtype=np.uint8)
    G = np.array(np.clip(Y - 0.34414 * (Cb-128) - 0.71414*(Cr-128),0,255), dtype=np.uint8)
    B = np.array(np.clip(Y + 1.772 * (Cb-128),0,255), dtype=np.uint8)

    img_RGB = np.empty((R.shape[0],R.shape[1],3), dtype= np.uint8)
    img_RGB[:,:,0] = R
    img_RGB[:,:,1] = G
    img_RGB[:,:,2] = B

    return img_RGB
