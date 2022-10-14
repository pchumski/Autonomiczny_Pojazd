import cv2
import numpy as np

cap = cv2.VideoCapture(0) # ładowanie obrazu z kamery z komputera
# ładowanie filmiku o podanej nazwie
# cap = cv2.VideoCapture('videname.mp4') 

while True: # nieskonczona petla while
    # odczytuje pokolei kazda klatke z obrazu z kamery
    ret, frame = cap.read()
    width = int(cap.get(3)) # pobieranie szerokosci z klatki kamery 
    height = int(cap.get(4)) # pobieranie wysokosci z klatki kamery 
    
    # Nowy obraz z narysowana niebieska linia 
    img = cv2.line(frame, (0,0), (width, height), (255,0,0), 10) # rysowanie lini na obrazie
    img = cv2.line(img, (0,height), (width, 0), (0,255,0), 10) # rysowanie lini na obrazie
    
    # Nowy obraz z kwadratem  
    img = cv2.rectangle(img, (100,100), (200,200), (128,128,128), 5) # Wartosc -1 w thickness daje wypelnienie
    
    # Nowy obraz z kołem
    img = cv2.circle(img, (500,250), 60, (0,0,255), -1) 
    
    # Dodawanie tekstu
    font = cv2.FONT_HERSHEY_SIMPLEX # dodawanie czcionki napisu
    img = cv2.putText(img, 'My camera frame', (20, height - 15), font, 1.5, (0,0,0), 3, cv2.LINE_AA )
    
    cv2.imshow('frame', img) # wyswietlanie obrazu ( dokladniej jednej klatki z kamery )
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q
        break
    
cap.release()
cv2.destroyAllWindows()