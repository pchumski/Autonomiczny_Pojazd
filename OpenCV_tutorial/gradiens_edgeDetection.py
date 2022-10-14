import cv2
import numpy as np
from matplotlib import pyplot as plt

# Ladujemy w skali szarosci zeby wykorzystac gradient
# img = cv2.imread('assets/soccer_practice.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('images/sudoku.PNG', cv2.IMREAD_GRAYSCALE)

#Laplace gradient
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# Absolute value laplacian
lap = np.uint8(np.absolute(lap))
# Sobel

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)


titles = ['image', 'Laplacian', 'Sobel X', 'Sobel Y', 'Sobel Combined']
images = [img, lap, sobelX, sobelY, sobelCombined]


for i in range(len(images)):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()