import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('assets/logo.jpg')

cv2.imshow('Image', img)

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Musimy zmienic z BGR na RGB bo tak czyta obraz Matplotlib
plt.imshow(img1)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()