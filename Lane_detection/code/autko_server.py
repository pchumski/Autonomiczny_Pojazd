from flask import Flask, render_template, Response, session, request
import cv2
import numpy as np
from my_detect import object_detection 
from datetime import timedelta
from utils.distance import stackImages
from Lane_detection import getLaneCurve
from autko import getLaneCurve
from jetracer.nvidia_racecar import NvidiaRacecar
import socket

#boxes = [[1,2,3,4,5,6,7,8]]
stop_flag = False
camera_flag = False

car.throttle = 0.0
car.throttle_gain = 0.3

car.steering_gain= 0.4
car.steering_offset = -0.2
car.steering = 0.0

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("192.168.1.13", 30003))
server_socket.listen()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=224,
    capture_height=224,
    display_width=224,
    display_height=224,
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
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    while True:
        client_socket, client_address = server_socket.accept()
        data = client_socket.recv(6)
        numbers = [b for b in bytearray(data)]
        ret_val,frame_org = cap.read()
        img_line = np.copy(frame_org)
        frame_c = np.copy(frame_org)
        #frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

        img, box, crop = object_detection(frame_c, frame_org)
            
        #boxes = box.copy()
#       img_line = cv2.resize(img_line,(480,240))
#       curve, img_l = getLaneCurve(img_line,display=2)
#       img = stackImages(1, [img, img_l])
        #img = cv2.resize(img_line,(320,320)) # RESIZE
        #curve,result = getLaneCurve(img)
        ret, buffer = cv2.imencode('.jpg', img)
            
        
        if numbers[2] == 1:
            cv2.imwrite('zd/screen'+str(i)+'.jpg', img)
            i = i+1
        
        car.throttle = (float(numbers[1]) - 127.5)/127.5
        car.steering = (float(numbers[0]) - 127.5)/127.5 
        #throotle = (float(numbers[1]) - 127.5)/127.5 # normalizacja
        pot = float(numbers[3])/255
        if abs(car.steering) <0.1:
            car.steering = 0.0
        if abs(car.throttle) <0.1:
            car.throttle = 0.0
        if numbers[4] == 1:
            break
        
        frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        keyCode = cv2.waitKey(30) & 0xFF
            
        # if keyCode == 27:
        #     car.throttle = 0.0
        #     car.steering = 0.0
        #     break

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    show_camera()
    app.run(host='192.168.1.13')

