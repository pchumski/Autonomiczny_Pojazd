from pickle import TRUE
import numpy as np
import cv2


def nothing(x): # funkcja callback, ktora zostanie wywolana jak tylko zmieni sie cos na tracbarze
    print(x)
# Fukcje callback ktore zmieniaja kolor danego kanalu na taki w jakiej pozycji jest trackbar
def blue(x):
    img[:,:,0] = x
def green(x):
    img[:,:,1] = x
def red(x):
    img[:,:,2] = x

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image') # tworzy obraz o danej nazwie na ktorym mozemy umieszzcac np tracbar

# Pierwsza metoda za pomoca funkcji callback
# cv2.createTrackbar('B', 'image', 0, 255, blue)
# cv2.createTrackbar('G', 'image', 0, 255, green)
# cv2.createTrackbar('R', 'image', 0, 255, red)

#Druga metoda:
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('R', 'image', 0, 255, nothing)

# Tworzenie switcha
switch = '0 : OFF\n 1: ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while TRUE:
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q
        break
    
    # Druga metoda:
    b = cv2.getTrackbarPos('B','image')
    g = cv2.getTrackbarPos('G','image')
    r = cv2.getTrackbarPos('R','image')
    # Pozycja switcha
    s = cv2.getTrackbarPos(switch, 'image')
    if s== 1:
        img[:] = (b,g,r)
    else:
        img[:] = 0
    
    
    
    
cv2.destroyAllWindows()