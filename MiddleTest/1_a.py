import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./MiddleTest/bridge.jpeg', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
shifted = np.fft.fftshift(f) # 將頻率 (0, 0) 位移至中心

amp = np.abs(shifted)


print(type(f))
print(f)
#print(type(shifted))
#print(shifted)


a = f.flatten()
b = np.sort(a)
size = f.size
#print(f.size)
#print(amp.size)
index = b[int(np.round(3*size/4))]
index = int(index)
print(index)
print(b[index])
f1 = np.where(f>b[index],f,0)#keeping only 1/4 largest coefficients
print(f1)
shifted1 = np.fft.fftshift(f1)

amp1 = np.abs(shifted1)
# print(amp)
# print(type(amp))
# a = amp.flatten()
# b = np.sort(a)
# size = amp.size
# print(a)
# print(b)
# index = b[int(np.round(3*size/4))]
# index = int(index)
# print(index)
# print(b[index])
# amp1 = np.where(amp>b[index],amp,0)
# print(amp1)


cv2.imshow('FFT 2D', amp1 / np.max(amp1) * 255)    # 頻域表示

inversed = np.fft.ifft2(f1).real.astype('uint8')  # 逆轉換
cv2.imshow('INVERSE FFT 2D', inversed)           
cv2.imwrite("./MiddleTest/ImageA.jpg",inversed)
cv2.waitKey(0)
cv2.destroyAllWindows()