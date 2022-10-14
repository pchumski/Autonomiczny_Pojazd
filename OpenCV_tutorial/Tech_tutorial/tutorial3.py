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
    
    # Wyswietlanie 4 obrazow z kamery w roznych pozycjach obroconych 
    image = np.zeros(frame.shape, np.uint8 ) # tworzymy macierz samych zer o rozmiarach klatki z kamery (frame.shape zwraca rozmiary macierzy obrazu)
    
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # zmniejszanie obrazu (klatki kamery) o polowe 
    
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # top left
    image[height//2:, width//2:] = smaller_frame # bottom left
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # top right 
    image[height//2:, :width//2] = smaller_frame # bottom right
    
    cv2.imshow('frame', image) # wyswietlanie obrazu ( dokladniej jednej klatki z kamery )
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q
        break
    
cap.release()
cv2.destroyAllWindows()