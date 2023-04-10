import cv2

import random
# w openCV pixel jest reprezentowany przez 3 wartosci BGR - blue green red (odwrotnosc RGB) np [0, 255, 0] - pixel zielony
# wartosc pixeli to od 0 czyli nic do 255 czyli wszystko
# img.shape daje nam wymiary obrazu oraz ilosc "kanalow", czyli reprezentacja pixela na ilu wartosciach (zazwyczaj 3 przy kolorowych BGR)

def random_pixel(img):
    """
    Generuje losowe wartosci pixeli dla 100 pierwszych wierszy po calej szerokosci obrazow
    Args:
        img : Obraz na ktorym dokonujemy zmian
    """    
    for i in range(100):
        for j in range(img.shape[1]):
            img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

def copy_part_of_image(img):
    """
    Kopiuje czesc obrazu od 500 do 700 pixela w wierszu i 600 do 900 w kolumnach do innej czesci tego samego obrazu
    od 100 do 300 w wierszach i 650 do 950 w kolumnach
    Args:
        img : Obraz na ktorym dokonujemy zmian
    """    
    tag = img[500:700, 600:900] 
    img[100:300, 650:950] = tag
            
img = cv2.imread('assets/logo.jpg', -1)

# random_pixel(img)

copy_part_of_image(img)

cv2.imshow('Image',  img)
cv2.waitKey(0)
cv2.destroyAllWindows()