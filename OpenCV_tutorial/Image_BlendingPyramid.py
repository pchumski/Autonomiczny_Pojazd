import cv2
import numpy as np

apple = cv2.imread('images/apple.jpg')
orange = cv2.imread('images/oran.jpg')

orange = cv2.resize(orange, (apple.shape[1], apple.shape[0]))
apple_orange = np.hstack((apple[:, :300], orange[:, 300: ])) # Skladanie polowek dwoch obrazow

# generate Gaussian Piramide
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

orange_copy = orange.copy()
gp_orange = [orange_copy]
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)
    
# Generate Laplacian piramide
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5,0,-1):
    gaussian_extended = cv2.pyrUp(gp_apple[i],dstsize = (gp_apple[i - 1].shape[1], gp_apple[i - 1].shape[0]))
    laplacian = cv2.subtract(gp_apple[i-1], gaussian_extended)
    lp_apple.append(laplacian)
    
orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5,0,-1):
    gaussian_extended = cv2.pyrUp(gp_orange[i],dstsize = (gp_orange[i - 1].shape[1], gp_orange[i - 1].shape[0]))
    laplacian = cv2.subtract(gp_orange[i-1], gaussian_extended)
    lp_orange.append(laplacian)
    
# Now add left and right halves of images in each level

apple_orange_piramid = []
n= 0
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    n+= 1
    cols, rows, channels = apple_lap.shape
    laplacian = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):cols]))
    apple_orange_piramid.append(laplacian)

# now reconstruct
apple_orange_reconstruct = apple_orange_piramid[0]

for i in range(1,6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.resize(apple_orange_reconstruct, (apple_orange_piramid[i].shape[1], apple_orange_piramid[i].shape[0]))
    apple_orange_reconstruct = cv2.add(apple_orange_piramid[i], apple_orange_reconstruct)

cv2.imshow('Apple', apple)
cv2.imshow('Orange', orange)
cv2.imshow('Apple and Orange', apple_orange)
cv2.imshow('Apple Orange reconstruct', apple_orange_reconstruct)

cv2.waitKey(0)
cv2.destroyAllWindows()