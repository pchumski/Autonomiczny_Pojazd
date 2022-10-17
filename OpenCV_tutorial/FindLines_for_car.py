import numpy as np
import cv2

def make_coordinates(image, line_parameters):
    """Oblicznie wzoru na prostą opisującą linię: y = slope*x + intercept i zwracanie koordynatow x i y
    Args:
        image (array): Obraz na którym rysujemy linie
        line_parameters (array): parametry [slope, intercept]
    Returns:
        array: koordynaty [x1,y1,x2,y2] do rysowania lini
    """    
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(img, lines):
    """Oblicznie średniej z wczesniej znalezionych linii aby były gładsze
    Args:
        img (array): Obraz, na którym będą rysowane linie
        lines (array): znalezione linie
    Returns:
        array: linie średnie z lewej i prawej strony
    """    
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(img, left_fit_average)
        right_line = make_coordinates(img, right_fit_average)
        return np.array([left_line, right_line])
    elif len(left_fit):
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(img, left_fit_average)
        return np.array([left_line])
    elif len(right_fit):
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(img, right_fit_average)
        return np.array([right_line])
            
            
def canny_detect(img):
    """Znajdywanie krawędzi za pomocą funkcji cv2.Canny()
    Args:
        img (array): Obraz, na którym szukamy krawędzi
    Returns:
        array: Obraz z znalezionymi krawędziami
    """    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Gray scale image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def roi(img):
    """Wyodrębnienie częsci obrazu, która nas interesuje (Region of intrest)
    Args:
        img (array): Obraz pełny z którego chcemy wziąć fragment
    Returns:
        array: Obraz z wyszczególnionym ROI (reszta czarna)
    """    
    width = img.shape[1]
    height = img.shape[0]
    poligons = np.array([[(0, height), (width/2,height/2),(width,height)]], np.int32)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,poligons, 255)
    region = cv2.bitwise_and(img, mask)
    return region


def draw_lines(img, lines):
    """Funkcja do naniesienia linii na obraz
    Args:
        img (array): Obraz, na którym rysujemy linie
        lines (array): macierz z koordynatami linii do narysowania
    Returns:
        array: Obraz z naniesionymi liniami
    """    
    lane_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(lane_img, (x1,y1), (x2,y2), (255,0,0), 8)
    return lane_img
    
# Wykrywanie linii na obrazie

# image = cv2.imread('images/road_image.jpg') # Zwykly obraz BGR
# copy_img = np.copy(image) # Kopia obrazu
# canny_image = canny_detect(copy_img)
# region_of_intrest = roi(canny_image)
# lines = cv2.HoughLinesP(region_of_intrest, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_lines = average_slope_intercept(copy_img, lines)
# lines_img = draw_lines(copy_img, average_lines)
# combo_img = cv2.addWeighted(copy_img, 0.8, lines_img, 1, 1)
# cv2.imshow('Image', combo_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Wykrywanie Lini na filmie

cap = cv2.VideoCapture('videos/test2.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        canny_image = canny_detect(frame)
        region_of_intrest = roi(canny_image)
        lines = cv2.HoughLinesP(region_of_intrest, 2, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=3)
        average_lines = average_slope_intercept(frame, lines)
        lines_img = draw_lines(frame, average_lines)
        combo_img = cv2.addWeighted(frame, 0.8, lines_img, 1, 1)
        cv2.imshow('Frame', combo_img)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()