import cv2
import numpy as np

cap = cv2.VideoCapture(0) 

# Uzyjemy pre-trenowanego klasyfikatora do wykrywania twarzy oraz oczu
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True: # nieskonczona petla while
    # odczytuje pokolei kazda klatke z obrazu z kamery
    ret, frame = cap.read()
    width = int(cap.get(3)) # pobieranie szerokosci z klatki kamery 
    height = int(cap.get(4)) # pobieranie wysokosci z klatki kamery 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Zeby wykryc twarz i oczy potzrebujemy obrazu w skali szarosci
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # zwraca prostokaty gdzie beda kandydaci do bycia twarza, a konkretnie polozenie lewgo dolnego rogu oraz szerokosc i wysokosc 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)
        roi_gray = gray[y:y+h, x:x+w] # roi - region of intrest ( wydobywamy z obrazu fragment gdzie wykryto twarz by wykryc w nim oczy, poniewaz wiemy ze oczy sa na twarzy)
        roi_color = frame[y:w+h, x:x+w] # Modyfikujac w ten sposob ten wycinek obrazu modyfikuje tez oryginalny obraz 
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 5)
            
    
    cv2.imshow('frame', frame) # wyswietlanie obrazu ( dokladniej jednej klatki z kamery )
    
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q
        break
    
cap.release()
cv2.destroyAllWindows()