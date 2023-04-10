import cv2
import numpy as np

img = cv2.imread('images/gradient.png', 0)

# Theresolding Binary
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # Jezeli pixele maja wieksza wartosc niz 127 i mniejsza niz 255 to zwraca 1 (czyli bialy), natomiast tam gdzie jest mniejszy od 127 i wiekszy niz 255(niemozliwe) zwraca 0 ( w tym konkretnym przyladzie)

_, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) # odwrocone dzialanie Theresolding binary

_, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC) # Zmienia pixele wieksze od 127 w 127

_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO) # Zmienia pixele mniejsze od 127 w 0, a reszte zostawia bez zmian
# Istnieje jeszcze THRESH_TOZERO_INV czyli odwrotnosc

# Adaptive Thresholding
# Obliczny thresolding nie jest globalny dla calego obrazu, tylko liczony jest osobno dla roznych miejsc w obrazie
img1 = cv2.imread('images/sudoku.PNG', 0)
_, th11 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
th21 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th31 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# cv2.imshow("image", img)
# cv2.imshow("Theresolding", th1)
# cv2.imshow("th2", th2)
# cv2.imshow("th3", th3)
# cv2.imshow("th4", th4)

cv2.imshow('Image1', img1)
cv2.imshow("th11", th11)
cv2.imshow("th21", th21)
cv2.imshow("th31", th31)


cv2.waitKey(0)
cv2.destroyAllWindows()