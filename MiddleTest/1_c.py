import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def dividefftC(img1):
    f1 = np.fft.fft2(img1)
    #f1 = np.fft.fft2(img_new)
    #print(f1)
    f1 = np.pad(f1, ((64,64),(64,64)), 'constant')#top,bottom.left.right
    #print(f1.size)
    #print(f1[0][:].size)
    #print(f1)
    
    inversed = np.fft.ifft2(f1).real.astype('uint8')      #逆轉換
    #cv2.imshow('INVERSE FFT 2D', inversed)
    return inversed

if (__name__=='__main__'):
    img = cv2.imread('./MiddleTest/bridge.jpeg', cv2.IMREAD_GRAYSCALE)
    img_new = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
    #print(img)
    #print(img_new)
    inversed_new = dividefftC(img_new)
    

    cv2.imshow("resize",inversed_new)
    cv2.imwrite("./MiddleTest/ImageC.jpg",inversed_new)

    cv2.waitKey(0)
    
