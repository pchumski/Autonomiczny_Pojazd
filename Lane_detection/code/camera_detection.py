import cv2

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

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
cap.set(3, 320)
cap.set(4, 320)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('jetracer_video.avi', fourcc, 30.0, (320, 320))

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
    cv2.imshow('JetRacer Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()