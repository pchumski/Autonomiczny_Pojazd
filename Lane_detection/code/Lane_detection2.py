import cv2
import numpy as np
import utlis
from flask import Flask, render_template, Response


app = Flask(__name__)
curveList = []
avgVal=10
 
def getLaneCurve(img,display=2):
 
    imgCopy = img.copy()
    imgResult = img.copy()
    #### STEP 1
    imgThres = utlis.thresholding(img)
 
    #### STEP 2
    hT, wT, c = img.shape
    points = np.float32([(58, 72), (wT-58, 72),(0 , hT ), (wT, hT)])
    imgWarp = utlis.warpImg(imgThres,points,wT,hT)
    imgWarpPoints = utlis.drawPoints(imgCopy,points)
 
    #### STEP 3
    middlePoint,imgHist = utlis.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint
 
    #### SETP 4
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
 
    #### STEP 5
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        #cv2.imshow('ImageStack', imgStacked)
        # ret, buffer = cv2.imencode('.jpg', imgStacked)
        # frame = buffer.tobytes()
        # yield(b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        curve = curve/100
        if curve>1: curve ==1
        if curve<-1:curve == -1
        return curve, imgStacked
    elif display == 1:
        #cv2.imshow('Resutlt', imgResult)
        curve = curve/100
        if curve>1: curve ==1
        if curve<-1:curve == -1
        return curve, imgResult
        
    #### NORMALIZATION
 
    
 
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(show_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

def show_camera():
    cap = cv2.VideoCapture(0)
    # intialTrackBarVals = [102, 80, 20, 214 ]
    # utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("Camera Frame",0) >= 0:
            frameCounter += 1
            if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frameCounter = 0
 
            success, img = cap.read()
            img = cv2.resize(img,(480,240))
            curve, img = getLaneCurve(img,display=2)
            
            ret, buffer = cv2.imencode('.jpg', img)
        
            frame = buffer.tobytes()
    
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #print(curve)
            keyCode = cv2.waitKey(30) & 0xFF
            
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("unable to open camera")
if __name__ == '__main__':
    show_camera()
    app.run(host='192.168.0.35')