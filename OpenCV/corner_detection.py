from turtle import color
import numpy as np
import cv2

# zanim zaczniemy wykrywac krawedzie musimy zamienic na obraz na czarno-bialy (Grayscale)
img = cv2.imread('assets/chessboard.png')
img = cv2.resize(img, (0,0), fx=0.75, fy=0.75) # zmiana rozmiaru obrazu
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # zamiana na obraz czarno bialy

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10) # szukanie 100 rogow z jakoscia 0.01 (od 0 do 1) i minimalnym odspetem miedzy 2 punktami 10
# nasze rogi(corners) sa typu float, a zeby je narysowac na obrazie potrzebujemy int
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel() # rozklada macierz na plaski wektor np. [[2,3,4]] -> [2,3,4] lub np. [[2,3], [1,2]] -> [2,3,1,2]
    cv2.circle(img, (x,y), 5, (255,0,0), -1) # rysowanie kola wokol znalezionych rogow
    
# rysowanie lini miedzy rogami  
# for i in range(len(corners)):
#     for j in range(i+1, len(corners)):
#         corner1 = tuple(corners[i][0])
#         corner2 = tuple(corners[j][0])
#         color = tuple(map(lambda x: int(x), np.random.randint(0,255, size=3)))
#         cv2.line(img, corner1, corner2, color, 1)
        

cv2.imshow('Chessboard', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
