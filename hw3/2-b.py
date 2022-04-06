import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#HSV-V

def equalize(src,p,f):
    #equalize
    m=512
    n=512
    N=m*n
    equalize = np.array([])
    for i in range(256):
        equalize=np.append(equalize,(255/N)*(sumf(i,f)/1.0))
    equalize = np.round(equalize,decimals=0)    
    #print("E:",equalize)
    #把均值化的結果套用進原圖
    test = src
    #print("Esrc:",test)
    #print(src.min())
    for i in range(int(src.min()),int(src.max())):
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

def p(nums):
    #原圖灰階0~255出現機率
    m=512
    n=512
    N=m*n
    p = np.array([])
    for i in range(256):
        p = np.append(p,nums[i]/(3*N))
        #R,G,B
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
    src = cv.imread("./mp2a.jpeg")
    if src is None:
        print('Could not open or find the image')
        exit(0)
    #H：色相
    #S：飽和度
    #V：明度
    src = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    h,s,v=cv.split(src)

    # f1 = f(h)
    # f2 = f(s)
    f3 = f(v)
    # #print(f1)
    # p1 = p(f1)
    # p2 = p(f2)
    p3 = p(f3)
    # #print(p1)
    # e1 = equalize(h,p1,f1)
    # e2 = equalize(s,p2,f2)
    e3 = equalize(v,p3,f3)



    result = cv.merge([h, s, e3])
    cv.imshow("result",result)
    cv.imwrite("HW3-2-b hsv-v Equalized Image.jpg",result)
    
    # print("H:")
    # print(h)
    # print("S:")
    # print(s)
    # print("V:")
    # print(v)

    dst1 = cv.equalizeHist(v)

    # dst2 = cv.equalizeHist(g)
    # dst3 = cv.equalizeHist(r)
    # dst = cv.merge([dst1,dst2,dst3])


    dst = cv.merge([h,s,dst1])
    
    #cv.imshow('Source image', src)
    #cv.imshow('Equalized Image', dst)
    #cv.imshow("result",result)
    #cv.imwrite("mp2a-hsv.jpeg",src)
    #cv.imwrite("HW3-2-b Opencv hsv-v Equalized Image.jpg",dst)

    # # #OpenCV
    cv.waitKey()
    