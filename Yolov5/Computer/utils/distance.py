import cv2
import torch
import numpy as np
from utils.general import non_max_suppression

know_distance = 30
known_width_sign = 7.8
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
