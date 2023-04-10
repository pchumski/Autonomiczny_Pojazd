import cv2
import numpy as np

curveList = []
avgVal = 10

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

def getHistogram(img, minPer=0.1, display=False, region=1):

    if region ==1:
        histValues = np.sum(img, axis=0)
    else :
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
    #histValues = np.sum(img, axis=0)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    indexArray =np.where(histValues >= minValue)
    basePoint =  int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint

def getLaneCurve(img):
    imgResult =img.copy()
    #STEP 1
    imgThres = thresholding(img)

    #STEP 2
    hT, wT, c = img.shape
    points = np.float32([(80, 177), (wT-80, 177),(40, 320 ), (wT-40, 320)])
    imgWarp = warpImg(imgThres, points, wT, hT)

    #STEP 3
    middlePoint,imgHist1 = getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint,imgHist2 = getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint-middlePoint

    #STEP 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    #STEP 5
    imgInvWarp = warpImg(imgWarp, points, wT, hT,inv = True)
    imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
    imgInvWarp[0:hT//3,0:wT] = 0,0,0
    imgLaneColor = np.zeros_like(img)
    imgLaneColor[:] = 0, 255, 0
    imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
    imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
    midY = 450
    cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
    cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
    cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
    for x in range(-30, 30):
        w = wT // 20
        cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
   
    # cv2.imshow('Resutlt',imgResult)


    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Hist1', imgHist1)
    cv2.imshow('Hist2', imgHist2)
    # print(curve)
    return curve,imgResult
    
    
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
        curve,imgResult = getLaneCurve(img)
        cv2.imshow('Video', img)
        cv2.imshow('Result', imgResult)
        if cv2.waitKey(1) == ord('q'):
            
            break
    cap.release()
    cv2.destroyAllWindows()

