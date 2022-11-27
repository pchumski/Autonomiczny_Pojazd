import cv2
import time
import os
import numpy as np
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

# cap.set(3, 320) # zmiena szerokosci
# cap.set(4, 320) # zmiana wysokosci
starting_time =time.time()
Frame_Counter= 0
Cap_frame =0 
Dir_name = "capture_images"
number_image_captured =20
capture_image=False

while True:
    IsDirExist = os.path.exists(Dir_name)
    print(IsDirExist)
    # if there is no Directory named "capture_image", simply create it. using os 
    if not IsDirExist:
        os.mkdir(Dir_name)
    Frame_Counter+=1
    
    ret, frame = cap.read()
    saving_frame = np.copy(frame)
    height, width, dim = saving_frame.shape
    
    cv2.putText(saving_frame, f"Height: {height}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0),1)
    cv2.putText(saving_frame, f"Width:  {width}", (30, 70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0),1)
    
    
    if capture_image==True and Cap_frame <= number_image_captured:
        Cap_frame+=1
        cv2.putText(frame, 'Capturing', (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,244, 255),1)
        cv2.imwrite(f"{Dir_name}/frame-{Cap_frame}.png", saving_frame)
    else:
        cv2.putText(frame, 'Not Capturing', (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0, 255),1)
        Cap_frame =0
        capture_image =False
    cv2.imshow("frame", frame)
    cv2.imshow("saving Image", saving_frame)
    total_time = time.time()
    frame_time=total_time -starting_time
    # calculating how much frame pass in each second

    fps = Frame_Counter/frame_time
    # print(fps)
    # print(capture_image)
     # when we press 'q' it quites the program
    if cv2.waitKey(1) == ord('q'):
        break
    # if we press 'c' on the keyboard then it starts capturing the images. 
    if cv2.waitKey(1)==ord('c'):
        capture_image= True
   
# finally closing the camera 
cap.release()
# closing all opend windows
cv2.destroyAllWindows()