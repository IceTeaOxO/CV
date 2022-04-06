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
        equalize=np.append(equalize,(255/N)*(sumf(i,f)/3))
    equalize = np.round(equalize,decimals=0)    
    print("E:",equalize)
    #把均值化的結果套用進原圖
    test = src
    print("Esrc:",test)
    print(src.min())
    for i in range(int(src.min()),int(src.max())):
        test[test==int(i)]=equalize[i]
    
    print("test",test[i])
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




    #f = f(src)
   # print(f)
    #print(f.max())
    #print(src.max())
    #print(src.min())
    #print(src[src==int(2)].size)
    #print(448*384)

    #p = p(f)
   # e = equalize(src,p,f)
    #print(e)
    #print(f[0])
    #cv.imshow("result",e)
    #cv.imwrite("HW3 Equalized Image.jpg",e)
    #cv.waitKey()


    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    #b,g,r=cv.split(src)

    img1 = src.copy()
    img1[:,:,1]=0
    img1[:,:,2]=0
    print(img1)
    f1 = f(img1)
    print(f1)
    p1 = p(f1)
    print(p1)

    sum = 0
    for i in range(256):
        sum+=p1[i]
    print(sum)
    
    e1 = equalize(img1,p1,f1)

    #print(img1)
    orin = np.array([])
    # for i in range(512):
    #     print(src[0][i])
    #     for j in range(512):
    #         orin = np.append(orin,src[i][j])
    #cv.imshow("B",img1) 
    cv.imshow("hist1",e1) 
    
    

    # img2 = src.copy()
    # img2[:,:,0]=0
    # img2[:,:,2]=0
    # cv.imshow("G",img2) 
    # print(img2)
    # f2 = f(img2)
    # p2 = p(f2)
    # e2 = equalize(img2,p2,f2)
    # cv.imshow("hist2",e2) 



    # img3 = src.copy()
    # img3[:,:,0]=0
    # img3[:,:,1]=0
    # cv.imshow("R",img3) 
    # print(img3)
    # f3 = f(img3)
    # p3 = p(f3)
    # e3 = equalize(img3,p3,f3)
    # cv.imshow("hist3",e3) 

    
    #dst = cv.equalizeHist(src)
    #print(src)

    
    #cv.imshow('Source image', src)
    #cv.imshow('Equalized Image', dst)
    #cv.imshow("result",result)
    #cv.imwrite("HW3-2-a Equalized Image.jpg",dst)

    # # #OpenCV
    cv.waitKey()