import cv2
import numpy as np

cap = cv2.VideoCapture('videos/ruch_drogowy.mp4') # otwarcie pliku video

# Odczytujemy dwie pierwsze klatki nagrania
ret, frame1 = cap.read()
ret, frame2 = cap.read()



while cap.isOpened():
    
    diff = cv2.absdiff(frame1, frame2) # oblicza aboslutna roznice miedzy dwoma klatkami nagrania
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # zmeiniamy w skale szarosci
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, (3,3), iterations=3)
    contours, hierarhy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(frame1, contours, -1, (0,255,0), 3)
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        
        if cv2.contourArea(c) < 900:
            continue
        else:
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
    
    cv2.imshow("Frame1", frame1)
    
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
