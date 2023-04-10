import cv2
import numpy as np

def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([45, 40, 148])
    upperWhite = np.array([155, 88, 255])
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

def getHistogram(img, minPer=0.1, display=False):
    histValues = np.sum(img, axis=0)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    indexArray =np.where(histValues >= minValue)
    basePoint =  int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint

def getLaneCurve(img):
    imgThres = thresholding(img)
    hT, wT, c = img.shape
    points = np.float32([(80, 177), (wT-80, 177),(40, 320 ), (wT-40, 320)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    basePoint,imgHist = getHistogram(imgWarp, display=True)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Hist', imgHist)
    
    
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

