import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def dividefft(img1):
    f1 = np.fft.fft2(img1)
    #shifted1 = np.fft.fftshift(f1)     #將頻率 (0, 0) 位移至中心

    a = f1.flatten()
    b = np.sort(a)
    size = f1.size
    index = b[int(np.round(3*size/4))]
    index = int(index)
    #print("25% largest",b[index])

    f1 = np.where(f1>b[index],f1,0)#keeping only 1/4 largest coefficients
    #shifted1 = np.fft.fftshift(f1)
    #amp1 = np.abs(shifted1)
    #cv2.imshow('FFT 2D', amp1 / np.max(amp1) * 255)       #頻域表示
    inversed = np.fft.ifft2(f1).real.astype('uint8')      #逆轉換
    #cv2.imshow('INVERSE FFT 2D', inversed)
    return inversed           

if (__name__=='__main__'):
    img = cv2.imread('./MiddleTest/bridge.jpeg', cv2.IMREAD_GRAYSCALE)
    img1 = img[0:16,0:255]
    img2 = img[16:32,0:255]
    img3 = img[32:48,0:255]
    img4 = img[48:64,0:255]
    img5 = img[64:80,0:255]
    img6 = img[80:96,0:255]
    img7 = img[96:112,0:255]
    img8 = img[112:128,0:255]
    img9 = img[128:144,0:255]
    img10 = img[144:160,0:255]
    img11 = img[160:176,0:255]
    img12 = img[176:192,0:255]
    img13 = img[192:208,0:255]
    img14 = img[208:224,0:255]
    img15 = img[224:240,0:255]
    img16 = img[240:255,0:255]

    inversed1 = dividefft(img1)
    inversed2 = dividefft(img2)
    inversed3 = dividefft(img3)
    inversed4 = dividefft(img4)
    inversed5 = dividefft(img5)
    inversed6 = dividefft(img6)
    inversed7 = dividefft(img7)
    inversed8 = dividefft(img8)
    inversed9 = dividefft(img9)
    inversed10 = dividefft(img10)
    inversed11 = dividefft(img11)
    inversed12 = dividefft(img12)
    inversed13 = dividefft(img13)
    inversed14 = dividefft(img14)
    inversed15 = dividefft(img15)
    inversed16 = dividefft(img16)
    inversed = img.copy()
    inversed[0:16,0:255] = inversed1
    inversed[16:32,0:255] = inversed2
    inversed[32:48,0:255] = inversed3
    inversed[48:64,0:255] = inversed4
    inversed[64:80,0:255] = inversed5
    inversed[80:96,0:255] = inversed6
    inversed[96:112,0:255] = inversed7
    inversed[112:128,0:255] = inversed8
    inversed[128:144,0:255] = inversed9
    inversed[144:160,0:255] = inversed10
    inversed[160:176,0:255] = inversed11
    inversed[176:192,0:255] = inversed12
    inversed[192:208,0:255] = inversed13
    inversed[208:224,0:255] = inversed14
    inversed[224:240,0:255] = inversed15
    inversed[240:255,0:255] = inversed16


    cv2.imshow("img",img)
    # cv2.imshow("img1",img1)
    # cv2.imshow("img2",img2)
    # cv2.imshow("img3",img3)
    # cv2.imshow("img4",img4)
    # cv2.imshow("img5",img5)
    # cv2.imshow("img6",img6)
    # cv2.imshow("img7",img7)
    # cv2.imshow("img8",img8)
    # cv2.imshow("img9",img9)
    # cv2.imshow("img10",img10)
    # cv2.imshow("img11",img11)
    # cv2.imshow("img12",img12)
    # cv2.imshow("img13",img13)
    # cv2.imshow("img14",img14)
    # cv2.imshow("img15",img15)
    # cv2.imshow("img16",img16)
    cv2.imshow("ImageB",inversed)
    cv2.imwrite("./MiddleTest/ImageB.jpg",inversed)
    cv2.waitKey(0)
