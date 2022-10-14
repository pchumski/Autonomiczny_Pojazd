import numpy as np
import cv2

img = cv2.imread('assets/chessboard.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
ret, thres = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarhy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # contours to lista wszystkich konturow na obrazie, jest to numpy array o (x,y) koordynatach
print("Number of contour = ", str(len(contours)))
print(contours[0])

# Rysowanie konturow
cv2.drawContours(img, contours, -1, (255,0,0), 5)


cv2.imshow('Color Image', img)
# cv2.imshow('Gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()