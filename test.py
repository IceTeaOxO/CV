import cv2
import numpy as np
import random

#img = cv2.imread("../computer_vision/0.jpg")
#print(img.shape)

img = np.empty((300,300,3),np.uint8)

for row in range(300):
    for col in range(300):
        img[row][col] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

cv2.imshow('img',img)
cv2.waitKey(0)