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
    
    # Rozpoznawanie kolorow wymaga przeksztalcenia obrazu BGR na HSV - Hue, Saturation ang Lightness/Brightness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # konwertujemy obraz z schematu kolorow BGR na HSV
    
    lower_blue = np.array([90,50,50])  # najnizsza (najjasniejsza) wartosc pixela niebieskiego
    upper_blue = np.array([130,255,255]) # najwyzsza (najciemniejsza) wartosc pixela niebieskiego
    
    #Tworzenie maski (spektrum pixeli miedzy lower_blue i upperblue)
    mask = cv2.inRange(hsv, lower_blue, upper_blue )
    
    # wydobywanie tylko pixeli z zakresu maski (w tym konkretnym przypadku niebieskiego)
    result = cv2.bitwise_and(frame, frame, mask=mask) # bitwise_and oznacza ze daje 1 czyli zostawia kolor pixela tylko jesli jest taki sam jak w src1 i src2 (w tym przypadku mamy
    # jeden ten sam obraz dlatego zawsze bedzie 1), ale porownuje tez czy te pixele znajduja sie w masce
    
    cv2.imshow('frame', result) # wyswietlanie obrazu ( dokladniej jednej klatki z kamery ) z zostawiona przez nas barwa (niebieski)
    # cv2.imshow('mask', mask) # wyswietlanie maski do porownania 
    
    # maska zawiera wartosci bitowe 0 lub 1 jezeli pixel nalezy do zakresu badz nie, dlatego obraz maski jest czarno-bialy
    
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q 
        break
    
cap.release()
cv2.destroyAllWindows()