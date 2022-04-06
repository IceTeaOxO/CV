import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#RGB
def equalize(src,p,f):
    #equalize
    m=512
    n=512
    N=m*n
    equalize = np.array([])
    for i in range(256):
        equalize=np.append(equalize,(255/N)*(sumf(i,f)/1.5))
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
    src = cv.imread("./mp2a.jpeg")#, cv.IMREAD_COLOR
    if src is None:
        print('Could not open or find the image')
        exit(0)
    #cv.imshow("result",e)
    #cv.imwrite("HW3 Equalized Image.jpg",e)

    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    b,g,r=cv.split(src)

    f1 = f(r)
    f2 = f(g)
    f3 = f(b)
    #print(f1)
    p1 = p(f1)
    p2 = p(f2)
    p3 = p(f3)
    #print(p1)
    e1 = equalize(r,p1,f1)
    e2 = equalize(g,p2,f2)
    e3 = equalize(b,p3,f3)



    result = cv.merge([e3, e2, e1])
    cv.imshow("result",result)
    cv.imwrite("HW3-2-a Equalized Image.jpg",result)

  
    

    
    # print("R:")
    # print(r)
    # print("G:")
    # print(g)
    # print("B:")
    # print(b)

    dst1 = cv.equalizeHist(b)
    dst2 = cv.equalizeHist(g)
    dst3 = cv.equalizeHist(r)
    dst = cv.merge([dst1,dst2,dst3])
    # #print(src)

    
    #cv.imshow('Source image', src)
    #cv.imshow('Equalized Image', dst)
    #cv.imshow("result",result)
    cv.imwrite("HW3-2-a Opencv Equalized Image.jpg",dst)

    # # #OpenCV
    cv.waitKey()