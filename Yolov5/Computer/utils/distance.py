import cv2
import torch
import numpy as np
from utils.general import non_max_suppression

know_distance = 30
known_width_sign = 5.1
known_width_traffic_light = 3.3
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


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
