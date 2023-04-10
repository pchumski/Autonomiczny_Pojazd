import cv2
import numpy as np


def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0, 0, 80])
    upperWhite = np.array([179, 50, 180])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def getLaneCurve(img):
    imgThres = thresholding(img)
    hT, wT, c = img.shape
    points = np.float32([(100, 175), (wT-100, 175),(55, 320 ), (wT-55, 320)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    
    
cap = cv2.VideoCapture('C:/Users/pawci/Desktop/air/testowanko/jetracer.avi')
frameCounter = 0

if __name__ == '__main__':
    
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        ret, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(320,320)) # RESIZE
        getLaneCurve(img)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            
            break
    cap.release()
    cv2.destroyAllWindows()

