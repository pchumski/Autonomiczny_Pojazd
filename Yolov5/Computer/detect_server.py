from flask import Flask, render_template, Response, session, request
import cv2
import numpy as np
from my_detect import object_detection 
from datetime import timedelta
from utils.distance import stackImages
from Lane_detection import getLaneCurve

boxes = [[1,2,3,4,5,6,7,8]]
stop_flag = False
camera_flag = False

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(show_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html', message="")


def show_diffault_image():
    img = cv2.imread("image/Ref_image.png")
    ret, buffer = cv2.imencode('.jpg', img)
    frame = buffer.tobytes()
    return frame
    
def show_camera():
    global boxes
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)
        # Sprawdzenie czy kamera jest otwarta
        while cv2.getWindowProperty("Camera Frame",0) >= 0:
            ret_val,frame_org = cap.read()
            img_line = np.copy(frame_org)
            frame_c = np.copy(frame_org)
            frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

            img, box, crop = object_detection(frame_c, frame_org)
            
            # boxes = box.copy()
            # img_line = cv2.resize(img_line,(480,240))
            # curve, img_l = getLaneCurve(img_line,display=2)
            #img = stackImages(0.7, [img, crop])
            ret, buffer = cv2.imencode('.jpg', img)
        
            # Formatowanie na bity
            frame = buffer.tobytes()
            # Przekazywanie klastki z kamery za kazdym wywolaniem
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # Wylaczanie kamery
            keyCode = cv2.waitKey(30) & 0xFF
            
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Nie mozna otworzyc kamery")
        
if __name__ == "__main__":
    show_camera()
    app.run(host='192.168.1.12')

