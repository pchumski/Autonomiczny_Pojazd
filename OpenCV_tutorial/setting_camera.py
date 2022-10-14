from datetime import datetime
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # szerokosc obrazu z kamery
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # wysokosc obrazu z kamery

cap.set(3, 1280) # zmiena szerokosci
cap.set(4, 720) # zmiana wysokosci


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Width= ' + str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) + ' Height= ' + str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        date = str(datetime.now())
        frame = cv2.putText(frame, text, (25, int(cap.get(4) - 25)), font, 1, (255,0,0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, date, (25, 25), font, 1, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
        
cap.release()
cv2.destroyAllWindows()