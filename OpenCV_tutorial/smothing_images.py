import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('assets/logo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Homogeous filter k = 1/(k_width * k_height) * ones((k_weight,k_height))
kernel = np.ones((5,5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)
# Blur
blur = cv2.blur(img, (5,5))
# Gaussian filter
gblur = cv2.GaussianBlur(img, (5,5), 0)


titles = ['image', '2D Convolution', 'Blur', 'Gaussian blur']
images = [img, dst, blur, gblur]


for i in range(len(images)):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()