import cv2

img = cv2.imread('images/cyberpunk.png', 1)
img = cv2.resize(img, (0,0), fx= 0.5, fy= 0.5) # Jezeli chce uzyc fx i fy do skalowania obrazu to musze w Tuple 2 argument zrobic na (0,0)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# zapis przerobionego zdjecia do innego pliku 
cv2.imwrite('new_image.png', img)

# -1, cv2.IMREAD_COLOR : Load a color image
# 0, cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode 
# 1, cv2.IMREAD_UNCHANGED : Loads image such including alpha channel


cv2.imshow('Image',img)

# Robimy to by miec pewnosc ze zamkniemy wszystkie okna
cv2.waitKey(0) #0 znaczy ze czeka nieskonczonosc czasu
cv2.destroyAllWindows() # niszczy wszystkie okna 
