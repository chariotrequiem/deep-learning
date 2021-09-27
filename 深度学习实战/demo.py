# 当前版本 ： python3.7.11
# 开发时间 ： 2021/9/27 15:54
import tensorflow as tf
import cv2 as cv

src = cv.imread("D:\\images\\14.jpg")
cv.imshow("opencv-python", src)
cv.waitKey()
cv.destroyAllWindows()
