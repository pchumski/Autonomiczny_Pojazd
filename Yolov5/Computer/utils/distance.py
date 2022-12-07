import cv2
import torch
import numpy as np
from utils.general import non_max_suppression

know_distance = 30
known_width = 7.8
def FocalLength(measured_distance, real_width, width_in_rf_image):
    """Obliczanie długości ogniskowej
    Args:
        measured_distance (float): Zmierzona odległość obiektu od obiektywu
        real_width (float): Prawdziwa szerokość obiektu
        width_in_rf_image (float): Szerokość obiektu według ramki Yolo

    Returns:
        float: Ogniskowa
    """    
    focal_length = (width_in_rf_image * measured_distance)/real_width
    return focal_length
def Distance_finder(Focal_length, real_width, width_in_rf):
    """Obliczanie odległości obiektu od kamery
    Args:
        Focal_length (float): Ogniskowa
        real_width (float): Prawdziwa szerokość obiektu
        width_in_rf (float): Wielkość według ramki Yolo
    Returns:
        float: Odległość
    """    
    distance = (real_width * Focal_length)/width_in_rf
    return distance
def Focal_Webcam(frame, model, device, names,img_width = 320, display = False):
    image_t = np.copy(frame)
    height, width = frame.shape[:2]
    new_height = int((((img_width/width)*height)//32)*32)
    frame = cv2.resize(frame, (img_width,new_height))
    img = torch.from_numpy(frame).to(device)
    img = img.permute(2, 0, 1).float().to(device) 
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.55, 0.45) 
    object_width = 0
    for i, det in enumerate(pred):
        if len(det): 
            for d in det: 
                x1 = int(d[0].item() * width/img_width)
                y1 = int(d[1].item() * height/new_height)
                x2 = int(d[2].item() * width/img_width)
                y2 = int(d[3].item()* height/new_height)
                conf = round(d[4].item(), 2)
                c = int(d[5].item())
                
                object_width = x2 - x1
                if display == True:
                    image_t = cv2.rectangle(image_t, (x1, y1), (x2, y2), (0,255,0), 1) # box
                    image_t = cv2.putText(image_t, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    if display == True:                
        return object_width, image_t
    else:
        return object_width