import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('track')
cv2.resizeWindow('track', 640, 480)
cv2.createTrackbar('hl1', 'track', 0,255, nothing)
cv2.createTrackbar('sl1', 'track', 0,255, nothing)
cv2.createTrackbar('vl1', 'track', 0,255, nothing)

cv2.createTrackbar('hu1', 'track', 0,255, nothing)
cv2.createTrackbar('su1', 'track', 0,255, nothing)
cv2.createTrackbar('vu1', 'track', 0,255, nothing)

cv2.createTrackbar('hl2', 'track', 0,255, nothing)
cv2.createTrackbar('sl2', 'track', 0,255, nothing)
cv2.createTrackbar('vl2', 'track', 0,255, nothing)

cv2.createTrackbar('hu2', 'track', 0,255, nothing)
cv2.createTrackbar('su2', 'track', 0,255, nothing)
cv2.createTrackbar('vu2', 'track', 0,255, nothing)



while True:
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hl1 = cv2.getTrackbarPos('hl1', 'track')
    sl1 = cv2.getTrackbarPos('sl1', 'track')
    vl1 = cv2.getTrackbarPos('vl1', 'track')
    
    hu1 = cv2.getTrackbarPos('hu1', 'track')
    su1 = cv2.getTrackbarPos('su1', 'track')
    vu1 = cv2.getTrackbarPos('vu1', 'track')
    
    hl2 = cv2.getTrackbarPos('hl2', 'track')
    sl2 = cv2.getTrackbarPos('sl2', 'track')
    vl2 = cv2.getTrackbarPos('vl2', 'track')
    
    hu2 = cv2.getTrackbarPos('hu2', 'track')
    su2 = cv2.getTrackbarPos('su2', 'track')
    vu2 = cv2.getTrackbarPos('vu2', 'track')
    
    mask1 = cv2.inRange(hsv, np.array([hl1,sl1,vl1]), np.array([hu1,su1,vu1]))
    
    mask2 = cv2.inRange(hsv, np.array([hl2,sl2,vl2]), np.array([hu2,su2,vu2]))
    
    masks = mask1 + mask2
    
    result = cv2.bitwise_and(frame, frame, mask=masks)
    
    cv2.imshow("Oryginal Frame", frame)
    cv2.imshow("Result frame", result)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
    