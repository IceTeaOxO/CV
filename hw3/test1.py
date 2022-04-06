import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


if (__name__=='__main__'):
    src = cv.imread("./mp2.jpeg")
    if src is None:
        print('Could not open or find the image')
        exit(0)
    #src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # cv.imshow('Origin',src)
    # cv.imshow('Test',test)
    # cv.waitKey()
    #
    #dst = cv.equalizeHist(src)

    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(src)
    
    
    #print("OOOOOOOO",src)
    for i in range(len(src)):
        print("seeeee:",dst[i])
        print("OOOOOOOO",src[i])
    

    # # #OpenCV
    #cv.waitKey()