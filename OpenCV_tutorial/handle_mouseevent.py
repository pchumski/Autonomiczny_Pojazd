import cv2
import numpy as np

# events = [ i for i in dir(cv2) if 'EVENT' in i] 
# wyswietlenie wszystkich eventow (dir daje wszystkie funkcje i zmienne opencv)

# print(events)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Wyswietalnie miejsca (x,y) gdzie sie kliknelo 
        print(x,',',y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ',' + str(y)
        cv2.putText(img, strXY, (x,y), font, 1, (255,0,255), 2)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        # Wyswietlanie koloru BGR w miejscu w ktorym sie kliknie
        blue = img[y,x,0]
        green = img[y,x,1]
        red = img[y,x,2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strCH = str(blue) + ',' + str(green) + ',' + str(red)
        cv2.putText(img, strCH, (x,y), font, 1, (0,255,255), 2)
        cv2.imshow('image', img)
    if event == cv2.EVENT_MBUTTONDOWN:
        # Rysowanie kola w miejscu klikniecia scrolla oraz laczenie ich za pomoca lini gdy istnieja wiecej niz 2
        cv2.circle(img, (x,y), 4, (0,255,0), -1)
        points.append((x,y))
        if len(points) >= 2:
            cv2.line(img, points[-1], points[-2], (0,255,0), 3)
        cv2.imshow('image', img)
        
        

img = cv2.imread('assets/logo.jpg', -1)
cv2.imshow('image', img)
points = []
cv2.setMouseCallback('image', click_event) # Funkcja ktora ustawia zdarzenia po klikniecu myszka  ustanowianym w funkcji click_event( nazwa obrazu jest wazna zeby sie zgadzala )
 
cv2.waitKey(0)
cv2.destroyAllWindows()