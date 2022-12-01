from models.experimental import attempt_load
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
import torch
import cv2

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ] # Klasy zwykłego Yolov5
names = ['speedlimit', 'stop', 'crosswalk', 'trafficlight'] # Klasy mojego wytrenowanego modelu

# Funkcja do obsługi CSI camera przez OpenCV
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

yolov5_file = r'yolov5n.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(yolov5_file, device=device)

def object_detection(frame):
    frame = cv2.resize(frame, (640,640))
    img = torch.from_numpy(frame)
    img = img.permute(2, 0, 1).float().to(device)  #convert to required shape based on index
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.20) # prediction, conf, iou
    # print(pred)
    detection_result = []
    for i, det in enumerate(pred):
        if len(det): 
            for d in det: # d = (x1, y1, x2, y2, conf, cls)
                x1 = int(d[0].item())
                y1 = int(d[1].item())
                x2 = int(d[2].item())
                y2 = int(d[3].item())
                conf = round(d[4].item(), 2)
                c = int(d[5].item())
                
                detected_name = names[c]

                # print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
                detection_result.append([x1, y1, x2, y2, conf, c])
                
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1) # box
                if c!=1: # if it is not head bbox, then write use putText
                    frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return frame, detection_result

if __name__ == "__main__":
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture(0)
    #cap.set(3, 640) # zmiena szerokosci
    #cap.set(4, 640) # zmiana wysokosci
    
    while True:
        ret,frame = cap.read()
        
        result, box = object_detection(frame)
        
        cv2.imshow("Result", result)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
