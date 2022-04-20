import numpy as np
import matplotlib.pyplot as plt
import cv2

t = 2              #取樣時間
sample_rate = 800 # 取樣率，每秒取幾個樣本

def signal(t, sample_rate):
    f = 10
    x = np.linspace(0, t, int(t * sample_rate), endpoint = False)
    return (np.sin(f * 2 * np.pi * x) +    
            np.sin(3 * f * 2 * np.pi * x) / 2 + 
            np.sin(4 * f * 2 * np.pi * x) / 5 +
            np.sin(8 * f * 2 * np.pi * x) / 3)

samples = signal(t, sample_rate)
sp = np.fft.fft(samples) 

freq = np.fft.fftfreq(samples.size, d = 1 / sample_rate)
amp = np.abs(sp)

ax = plt.gca()
ax.stem(freq, amp / np.max(amp))
ax.set_xlim([np.min(freq), np.max(freq)])
plt.show()
#傅里叶变换之后，横坐标即为分离出的正弦信号的频率，纵坐标对应的是加权密度
#对图片的数据做傅立叶，然后增大高频信号的系数就可以提高图像的对比度。同样，相机自动对焦就是通过找图像的高频分量最大的时候，就是对好了。

#演算法步驟如下：

# 檢查影像的尺寸，如果不是 2 的整數冪則直接退出。
# 對影像的灰度值進行歸一化。
# 對影像的每一行執行一維 FFT，並儲存為中間結果。
# 對上一步結果中的每一列執行一維 FFT，返回變換結果。
# 將零頻分量移到頻譜中心，並求絕對值進行視覺化。
# 對中心化後的結果進行對數變換，以改善視覺效果。

def fft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]
    
    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(n//2) / n)
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X

def fft2(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    
    img = cv2.normalize(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(img[i, :])
    for j in range(w):
        res[:, j] = fft(res[:, j])
    return res

def fftshift(img):
    #swap the first and third quadrants, and the second and fourth quadrants
    h, w = img.shape
    h_mid, w_mid = h//2, w//2
    res = np.zeros([h, w], 'complex128')
    res[:h_mid, :w_mid] = img[h_mid:, w_mid:]
    res[:h_mid, w_mid:] = img[h_mid:, :w_mid]
    res[h_mid:, :w_mid] = img[:h_mid, w_mid:]
    res[h_mid:, w_mid:] = img[:h_mid, :w_mid]
    return res


def signal(t, sample_rate):
    f = 10
    x = np.linspace(0, t, int(t * sample_rate), endpoint = False)
    return (np.sin(f * 2 * np.pi * x) +    
            np.sin(3 * f * 2 * np.pi * x) / 2 + 
            np.sin(4 * f * 2 * np.pi * x) / 5 +
            np.sin(8 * f * 2 * np.pi * x) / 3)

samples = signal(t, sample_rate)
sp = np.fft.fft(samples) 

freq = np.fft.fftfreq(samples.size, d = 1 / sample_rate)
amp = np.abs(sp)

ax = plt.gca()
ax.stem(freq, amp / np.max(amp))
ax.set_xlim([np.min(freq), np.max(freq)])
plt.show()