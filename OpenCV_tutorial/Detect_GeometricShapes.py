import numpy as np
import cv2

def nothing(x):
    pass
img = cv2.imread('images/Geometric.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('tracbar') 
# cv2.createTrackbar('Thresolding', 'tracbar', 0, 255, nothing )
# cv2.createTrackbar('epsilon', 'tracbar', 0, 200, nothing)

# th1 = cv2.getTrackbarPos('Thresolding', 'tracbar')
# epsilon = cv2.getTrackbarPos('epsilon', 'tracbar')
_, thres = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
for contur in contours:
    approx = cv2.approxPolyDP(contur, 0.01*cv2.arcLength(contur, True), True)
    cv2.drawContours(img, [approx], 0, (0,0,0), 3)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
        
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(img, "square", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        else:
            cv2.putText(img, "rectangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 5:
        cv2.putText(img, "pentagon", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 6:
        cv2.putText(img, "hexagon", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 7:
        cv2.putText(img, "heptagon", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 8:
        cv2.putText(img, "octagon", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    elif len(approx) == 10:
        cv2.putText(img, "star", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    else:
        cv2.putText(img, "circle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        
cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()