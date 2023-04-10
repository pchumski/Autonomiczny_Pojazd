import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/kolorowe-kulki.jpg', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=2)
erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closeing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']
images = [img, mask, dilation, erosion, opening, closeing]

for i in range(len(images)):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()