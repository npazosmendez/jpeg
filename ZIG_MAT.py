import jpeg as jpeg
import numpy as np

MAT = np.zeros((8,8),np.int)
actual_val = 0
for i in range(8):
    for j in range(8):
        MAT[i,j] = actual_val
        actual_val += 1
print([MAT])

print("Zig zagging")
print(jpeg.fast_ZZPACK(np.array([MAT])))

print("UN-Zig zagging")
print(jpeg.zig_zag_unpacking(jpeg.fast_ZZPACK(np.array([MAT]))))
        

ZIG_MAT = [[0,8,1,2,9,16,24,17],[10,3,4,11,18,25,32,40],[33,26,19,12,5,6,13,20],[27\
    ,34,41,48,56,49,42,35],[28,21,14,7,15,22,29,36],[43,50,57,58,51,44,37,30],[23,31\
    ,38,45,52,59,60,53],[46,39,47,54,61,62,55,63]]

print(ZIG_MAT)


