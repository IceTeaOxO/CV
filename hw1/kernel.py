import cv2
import numpy as np
import random


kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
img = np.empty((100,100,1),np.uint8)
test=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
ker = np.array([0.5])
print(type(test))
a = np.array(test)
print(type(a))

b=np.pad(a,1,mode='constant')
print(b)
#print(test[1][1])
#2
#[[1,2,3,4,5],
#[1,2,3,4,5],
#[1,2,3,4,5],
#[1,2,3,4,5],
#[1,2,3,4,5]]
print(len(a))

for row in range(len(a)):
    x = row+1
    for elem in range(len(a[row])):
        y = elem+1
        b[x][y] = b[x-1][y-1]*kernel[0][0]+\
                b[x-1][y]*kernel[0][1]+\
                    b[x-1][y+1]*kernel[0][2]+\
                        b[x][y-1]*kernel[1][0]+\
                            b[x][y]*kernel[1][1]+\
                                b[x][y+1]*kernel[1][2]+\
                                    b[x+1][y-1]*kernel[2][0]+\
                                        b[x+1][y]*kernel[2][1]+\
                                            b[x+1][y+1]*kernel[2][2]

print(b)
        #print(x,y)
        



