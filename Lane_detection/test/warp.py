import cv2
import numpy as np

def empty(a):
    pass

def initializeTrackbars(intialTracbarVals,wT=320, hT=320):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, empty)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, empty)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, empty)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, empty)

def valTrackbars(wT=320, hT=320):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp


def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([45, 40, 148])
    upperWhite = np.array([155, 88, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite

def getLaneCurve(img):
    imgCopy = img.copy()
    imgThres = thresholding(img)
    hT, wT, c = img.shape
    points = valTrackbars()
    imgWarp = warpImg(img, points, wT, hT)
    imgWarpPoints = drawPoints(imgCopy, points)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('WarpPoints', imgWarpPoints)


frameCounter = 0

if __name__ == '__main__':
    cap = cv2.VideoCapture('C:/Users/pawci/Desktop/air/testowanko/jetracer.avi')
    intialTracbarVals = [100,175,55,320]
    initializeTrackbars(intialTracbarVals)
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

