import cv2
import numpy as np

img = cv2.imread('assets/logo.jpg')

# Gaussian Piramide
# lr = cv2.pyrDown(img)
# lr1 = cv2.pyrDown(lr)
layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    # cv2.imshow(str(i), layer)

#Laplacian Piramide
layer = gp[5]
cv2.imshow('upper level Gaussian Pyramid', layer)
lp = [layer]

for i in range(5,0,-1):
    gaussian_extended = cv2.pyrUp(gp[i],dstsize = (gp[i - 1].shape[1], gp[i - 1].shape[0]))
    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)
    
cv2.imshow('Oryginal Image', img)
# cv2.imshow('Pyramide Down Image1', lr)
# cv2.imshow('Pyramide Down Image2', lr1)

cv2.waitKey(0)
cv2.destroyAllWindows()