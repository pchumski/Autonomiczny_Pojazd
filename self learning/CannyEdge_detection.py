import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

cv2.namedWindow('Trackbar')
cv2.createTrackbar('Th1', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('Th2', 'Trackbar', 0, 255, nothing)

while True:
    img = cv2.imread('assets/soccer_practice.jpg', cv2.IMREAD_GRAYSCALE)
    th1 = cv2.getTrackbarPos('Th1', 'Trackbar')
    th2 = cv2.getTrackbarPos('Th2', 'Trackbar')
    
    canny = cv2.Canny(img, th1, th2)
    
    img = cv2.resize(img, (0,0),  fx=0.5, fy=0.5)
    canny = cv2.resize(canny, (0,0),  fx=0.5, fy=0.5)
    
    cv2.imshow('Image', img)
    cv2.imshow('Canny', canny)

    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()

    



