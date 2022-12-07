import torch
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)
#print(torch.cuda.is_available())
font = cv2.FONT_HERSHEY_SIMPLEX


model = torch.hub.load('yolov5', 'custom', 'best.pt', source='local')
model.cuda()

cap = cv2.VideoCapture(0)
cap.set(3, 640) # zmiena szerokosci
cap.set(4, 640) # zmiana wysokosci

def show_camera():

    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera Frame", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("Camera Frame",0) >= 0:
            ret_val,img = cap.read()
        
            framec = np.copy(img)
            framec = cv2.cvtColor(framec, cv2.COLOR_BGR2RGB)

            results = model(framec, size=640)

            #results.print()

            #print(results.pandas().xyxy)

            for i in range(len(results.pandas().xyxy[0])):
                wynik = results.pandas().xyxy[0]['confidence'][i]
                if float(wynik) >= 0.4:
                    xmin = results.pandas().xyxy[0]['xmin'][i]
                    ymin = results.pandas().xyxy[0]['ymin'][i]
                    xmax = results.pandas().xyxy[0]['xmax'][i]
                    ymax = results.pandas().xyxy[0]['ymax'][i]
                    klasa = results.pandas().xyxy[0]['name'][i]
                    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
                    cv2.putText(img, str(klasa), (int(xmin), int(ymin)-35), font, 1, (0,0,0), 1)
                    cv2.putText(img, str(round(wynik,2)), (int(xmin), int(ymin)-15), font, 1, (0,0,0), 1)
            #cv2.imshow("Frame", frame)
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
    app.run(host='192.168.1.12')