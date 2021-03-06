import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def equalize(src,p,f):
    #equalize
    m=448
    n=384
    N=m*n
    equalize = np.array([])
    for i in range(256):
        equalize=np.append(equalize,(255/N)*(sumf(i,f)/3))
    equalize = np.round(equalize,decimals=0)    
    print("E:",equalize)

    #把均值化的結果套用進原圖
    test = src
    for i in range(int(src.max())):
        test[test==int(i)]=equalize[i]
    

    #print("test",test[i])

    return test

def sumf(index,f):
    sum = 0
    for i in range(index):
        sum+=f[i]
    return sum



def cdf(index,p):
    #原圖灰階0~255累積出現機率
    cdf=0
    for i in range(index):
        cdf += p[i]
        #cdf = i*p[i]
    #print(cdf)
    return cdf


    #  for row in range(len(img)):
    #      for elem in range(len(img[row])):
    #          print(elem)

def p(nums):
    #原圖灰階0~255出現機率
    m=448
    n=384
    N=m*n
    p = np.array([])
    for i in range(256):
        p = np.append(p,nums[i]/N)
    #print("p:",p)
    return p

def f(src):
    #原圖灰階0~255出現次數
    data=np.array([])
    for row in range(len(src)):
         data = np.append(data,src[row])
        
    sorted_data = np.sort(data)
    nums = np.array([])

    for i in range(256):
        x = sorted_data==int(i)
        num = sorted_data[x].size
        nums=np.append(nums,num)
    #print("nums:",nums)
    return nums

if (__name__=='__main__'):
    src = cv.imread("./mp2.jpeg")
    if src is None:
        print('Could not open or find the image')
        exit(0)
    f = f(src)
    #print(f)
    #print(f.max())
    #print(src.max())
    #print(src.min())
    #print(src[src==int(2)].size)
    #print(448*384)

    p = p(f)
    e = equalize(src,p,f)
    #print(e)
    #print(f[0])
    cv.imshow("result",e)
    cv.imwrite("HW3-1 Equalized Image.jpg",e)
    cv.waitKey()
    
    #src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #dst = cv.equalizeHist(src)
    
    
    #cv.imshow('Source image', src)
    #cv.imshow('Equalized Image', dst)
    #cv.imshow("result",result)
    #cv.imwrite("HW3-1 OpenCV Equalized Image.jpg",dst)

    # # #OpenCV
    #cv.waitKey()
    

