from pickle import TRUE
import numpy as np
import cv2

def nothing(x):
    pass

# Tworzymy trackbary
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)



while True:
    frame = cv2.imread('images/kolorowe-kulki.jpg')
    # Zeby wykryc kolor trzeba zmienic obraz z BGR na HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")
    
    l_b = np.array([l_h, l_s, l_v]) # Kolor niebieski w HSV (dolny zakres)
    u_b = np.array([u_h, u_s, u_v]) # Kolor niebieski w HSV (gorny zakres zakres)
    
    # Tworzymy maske dla naszego obrazu
    mask = cv2.inRange(hsv_frame, l_b, u_b)
    
    # Porownujemy obraz z samym soba i maska
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", res)
    
    if cv2.waitKey(1) == ord('q'): # ord('q') - kod ASCI q
        break

cv2.destroyAllWindows()