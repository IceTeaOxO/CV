import cv2
import numpy as np

src_img = cv2.imread("../computer_vision/0.jpg")

kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

result_img = cv2.filter2D(src_img,-1,kernel)
cv2.imshow("origin",src_img)
cv2.imshow("filter2D img",result_img)
cv2.imwrite("Filter2d Sharpened Image.jpg",result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()