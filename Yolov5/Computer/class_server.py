from flask import Flask, render_template, Response, session, request
import cv2
import numpy as np
from my_detect import object_detection 
from datetime import timedelta

boxes = [[1,2,3,4,5,6,7,8]]
class Sign_detection():
    def __init__(self, cap):
        self.cap = cap
        self.boxes = list(list())
    def show_camera(self):
        if self.cap.isOpened():
            window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)

            while cv2.getWindowProperty("Camera Frame",0) >= 0:
                ret_val,frame_org = self.cap.read()
        
                frame_c = np.copy(frame_org)
                frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

                img, box = object_detection(frame_c, frame_org)
                if len(box):
                    #print(box)
                    self.boxes = box.copy()
                ret, buffer = cv2.imencode('.jpg', img)
        
                frame = buffer.tobytes()
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                keyCode = cv2.waitKey(30) & 0xFF
            
                if keyCode == 27:
                    break

            self.cap.release()
            cv2.destroyAllWindows()
        else:
            print("unable to open camera")
    def return_box(self):
        return self.boxes

app = Flask(__name__)
# app.secret_key = "hello"
# app.permanent_session_lifetime = timedelta(minutes=5)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    sign = Sign_detection(cv2.VideoCapture(0))
    return Response(sign.show_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/boxes')
def boxes():
    sign = Sign_detection(cv2.VideoCapture(0))
    return Response(sign.return_box())


if __name__ == "__main__":
    sign = Sign_detection(cv2.VideoCapture(0))
    sign.show_camera()
    app.run(host='192.168.1.12')

