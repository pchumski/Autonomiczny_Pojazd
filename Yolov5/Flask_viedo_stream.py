import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

def show_camera():
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)
        
        while cv2.getWindowProperty("Camera Frame",0) >= 0:
            ret_val, img = cap.read()
            
            ret, buffer = cv2.imencode('.jpg', img)
            
            frame = buffer.tobytes()
            
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            keyCode = cv2.waitKey(30) & 0xFF
            
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("unable to open camera")
        
        
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(show_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    show_camera()
    app.run(host='localhost')
        
