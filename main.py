import cv2
import numpy as np

print("Package Imported")

img = cv2.imread("Resources/KJ.jpg")
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
imgCanny = cv2.Canny(img, 150, 100)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imwrite("Resources/grayKJ.jpg", imgGray)
cv2.imwrite("Resources/blurKJ.jpg", imgBlur)
cv2.imwrite("Resources/CannyKJ.jpg", imgCanny)

cv2.imshow("Gray KJ", imgGray)

cv2.waitKey(0)
