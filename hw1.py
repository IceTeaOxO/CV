
from statistics import mode
import cv2
import numpy as np
import random


def filter(img,kernel):
    img_result = img
    image = np.pad(img,1,mode='constant')

    for row in range(len(img)):
        x = row+1
        for elem in range(len(img[row])):
            y = elem+1
            img_result[x-1][y-1] = image[x-1][y-1]*kernel[0][0]+\
                image[x-1][y]*kernel[0][1]+\
                    image[x-1][y+1]*kernel[0][2]+\
                        image[x][y-1]*kernel[1][0]+\
                            image[x][y]*kernel[1][1]+\
                                image[x][y+1]*kernel[1][2]+\
                                    image[x+1][y-1]*kernel[2][0]+\
                                        image[x+1][y]*kernel[2][1]+\
                                            image[x+1][y+1]*kernel[2][2]


    #print(image.shape)
    cv2.imshow('result',img_result)
    cv2.imshow("origin",image)
    print(img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(__name__=='__main__'):
    img = cv2.imread("../computer_vision/0.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #src_img = cv2.imread("../computer_vision/0.jpg")
    #'''
    kernel1 = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])
    #print(type(img))
    filter(img,kernel1)
    #'''
    '''
    kernel2 = np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ])
    filter(img,kernel2)
    '''
    '''
    kernel3 = np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ])
    filter(img,kernel3)
    '''