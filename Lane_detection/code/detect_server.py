from flask import Flask, render_template, Response, session, request
import cv2
import numpy as np
from my_detect import object_detection 
from datetime import timedelta
from utils.distance import stackImages
from Lane_detection import getLaneCurve
from autko import getLaneCurve
from jetcam.csi_camera import CSICamera

#boxes = [[1,2,3,4,5,6,7,8]]
stop_flag = False
camera_flag = False


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=320,
    display_height=320,
    framerate=120,
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






app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(show_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html', message="")

# @app.route("/stop/", methods=["POST"])
# def stop():
#     message = "STOP"
#     return render_template("index.html", message=message)
# @app.route('/SomeFunction')
# def SomeFunction():
#     print('In SomeFunction')
#     return "Nothing"


def show_diffault_image():
    img = cv2.imread("image/Ref_image.png")
    ret, buffer = cv2.imencode('.jpg', img)
    frame = buffer.tobytes()
    return frame
    
def show_camera():
    #global boxes
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap = CSICamera(width=320, height=320, capture_width=1080, capture_height=720, capture_fps=60)
    #if cap.isOpened():
        #window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)

    while True:
        #ret_val,frame_org = cap.read()
        frame_org = cap.read()
        img_line = np.copy(frame_org)
        frame_c = np.copy(frame_org)
        frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

        img, box, crop = object_detection(frame_c, frame_org)
            
            #boxes = box.copy()
#             img_line = cv2.resize(img_line,(480,240))
#             curve, img_l = getLaneCurve(img_line,display=2)
#             img = stackImages(1, [img, img_l])
            #img = cv2.resize(img_line,(320,320)) # RESIZE
            #curve,result = getLaneCurve(img)
        ret, buffer = cv2.imencode('.jpg', img)
            
            #car.steering_gain = 0.2
            #car.steering_offset = -0.1
            #car.steering = curve * car.steering_gain
        
            #car.steering = z
            #car.throttle = -0.4
            #car.throttle_gain = 0.4
        
        frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        keyCode = cv2.waitKey(30) & 0xFF
            
            #if keyCode == 27:
                #car.throttle = 0.0
                #car.steering = 0.0
                #break

    cap.release()
    cv2.destroyAllWindows()
    #else:
        #print("unable to open camera")
        
if __name__ == "__main__":
    show_camera()
    app.run(host='192.168.1.13')

