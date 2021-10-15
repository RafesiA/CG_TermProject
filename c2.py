import cv2
import numpy as np

img = cv2.imread("Resources/test.jpg")
print(img.shape)

imgResize = cv2.resize(img,(300, 500))
print(imgResize.shape)

imgCropped = img[0:600, 100:600]

cv2.imshow("Original", img)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)