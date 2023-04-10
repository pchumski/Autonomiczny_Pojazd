import numpy as np
import cv2


img = cv2.imread('assets/logo.jpg', -1)

# Tworzenie obrazu uzywajac numpy
# img = np.zeros([1024,1024,3], np.uint8)

img = cv2.line(img, (0,0), (255,255), (255,0,0), 5)
img = cv2.arrowedLine(img, (300, 50), (344,344), (0,255,0), 6)
img = cv2.rectangle(img, (500,500), (700,700), (120,55,255), -1)
img = cv2.circle(img, (300,650), 65, (123,123,45), 4)

# Text
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, "Hello World", (200, 900), font, 3, (0,0,0), 3)



cv2.imshow('Logo',img)


cv2.waitKey(0)
cv2.destroyAllWindows()