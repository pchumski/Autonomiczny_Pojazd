import cv2
import numpy as np
from jetracer.nvidia_racecar import NvidiaRacecar

curveList = []
avgVal = 10
car = NvidiaRacecar()

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
            cv2.line(imgHist,(int(x),img.shape[0]),(int(x),img.shape[0]-int(intensity//255//region)),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint

def getLaneCurve(img):
    imgResult =img.copy()
    #STEP 1
    imgThres = thresholding(img)

    #STEP 2
    hT, wT, c = img.shape
    points = np.float32([(100, 175), (wT-100, 175),(55, 320 ), (wT-55, 320)])
    imgWarp = warpImg(imgThres, points, wT, hT)

    #STEP 3
    middlePoint,imgHist = getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint,imgHist = getHistogram(imgWarp, display=True, minPer=0.9)
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


    # cv2.imshow('Thres', imgThres)
    # cv2.imshow('Warp', imgWarp)
    # cv2.imshow('Hist', imgHist)
    # print(curve)
    return curve,imgResult

def steering_control(angle):
    steering = (angle * 0.012)
    return steering
    
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=320,
    capture_height=320,
    display_width=320,
    display_height=320,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    

#frameCounter = 0

if __name__ == '__main__':
    #cap = cv2.VideoCapture('C:/Users/pawci/Desktop/air/testowanko/jet.avi')
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    while True:
        # frameCounter += 1
        # if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     frameCounter = 0

        ret, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(320,320)) # RESIZE
        curve,result = getLaneCurve(img)
        cv2.imshow('Video', result)

        car.steering_gain = 0.75
        car.steering_offset = -0.3
        car.steering = steering_control(curve)
        car.throttle = -0.26
        car.throttle_gain = 0.26

        if cv2.waitKey(1) == ord('q'):
            car.throttle = -0.01
            car.steering = -0.01
            break
    cap.release()
    cv2.destroyAllWindows()

