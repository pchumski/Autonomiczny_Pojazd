import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('assets/soccer_practice.jpg')

b,g,r = cv2.split(img)

cv2.imshow("img", img)
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

# hist = cv2.calcHist([img], [0], None, [256], [0, 256] )
# plt.plot(hist)


# plt.hist(img.ravel(), 256, [0, 256])# ravel rozklada macierz na "plaski" jednowierszowy wektor
plt.hist(b.ravel(), 256, [0, 256], color="blue")
plt.hist(g.ravel(), 256, [0, 256], color="green")
plt.hist(r.ravel(), 256, [0, 256], color="red")
plt.title("Histogram")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()