import numpy as np
import cv2

img = cv2.imread('assets/logo.jpg')
img2 = cv2.imread('assets/soccer_practice.jpg')

print(img.shape) # Zwraca rozmiar obrazu i liczbe kanalow
print(img.size) # Zwraca liczbe pixeli
print(img.dtype) # zwraca image datatype ktory jest otwarty 

b,g,r = cv2.split(img)

img = cv2.merge((b,g,r))

img = cv2.resize(img, (512,512))
img2 = cv2.resize(img2, (512,512))

dst1 = cv2.add(img, img2) # "dodawanie" dwoch obrazow, a bardziej ich przenikanie, nachodzenie na siebie (musza byc tych samych rozmiarow)

dst = cv2.addWeighted(img2,.3, img, .7, 0)

cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()