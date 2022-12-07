from models.experimental import attempt_load
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
import torch
import cv2
import numpy as np
from utils.distance import Distance_finder, know_distance,known_width


names = ['speedlimit', 'stop', 'crosswalk', 'trafficlight']

colors = list(np.random.rand(len(names),3)*255)
# 529
yolov5_file = r'bestn.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(yolov5_file, device=device)

focal_length = 529


def object_detection(frame, frame_t):
    height, width = frame.shape[:2]
    new_height = int((((320/width)*height)//32)*32)
    frame = cv2.resize(frame, (320,new_height))
    img = torch.from_numpy(frame).to(device)
    img = img.permute(2, 0, 1).float().to(device)  
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.45, 0.45) 
   
    detection_result = []
    for i, det in enumerate(pred):
        if len(det): 
            for d in det: 
                x1 = int(d[0].item() * width/320)
                y1 = int(d[1].item() * height/new_height)
                x2 = int(d[2].item() * width/320)
                y2 = int(d[3].item()* height/new_height)
                conf = round(d[4].item(), 2)
                c = int(d[5].item())
                
                object_width = x2 - x1
                distance = Distance_finder(known_width, focal_length, object_width)
                
                detected_name = names[c]

                # print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
                detection_result.append([x1, y1, x2, y2, conf, c])
                
                frame_t = cv2.rectangle(frame_t, (x1, y1), (x2, y2), colors[c], 1) # box
                frame_t = cv2.putText(frame_t, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c], 1, cv2.LINE_AA)
                frame_t = cv2.putText(frame_t, f'Distance: {round(distance,2)}', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c], 1, cv2.LINE_AA) 

    return frame_t, detection_result

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,frame = cap.read()
        frame_c = np.copy(frame)
        result, box = object_detection(frame_c, frame)
        
        cv2.imshow("Result", result)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
