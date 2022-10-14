import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    """Funkcja do trackbara, ktora nic nie robi
    Args:
        x (any): zmienna trackbara
    """    
    pass
def region_of_intrest(img, vertices):
    """Funkcja, która zostawia na obrazie tylko miejsce zainteresowania ROI
    Args:
        img (array): Obraz, który chcemy przetworzyć
        vertices (array): macierz z wyszególnionym miejscem zainteresowania ROI

    Returns:
        array: Obraz, który pokazuje tylko ROI
    """    
    mask = np.zeros_like(img) # tworzy macierz zer rozmiarow obrazu
    if len(img.shape) == 3:
        channel_count = img.shape[2] # liczba kanalow obrazu
    else:
        channel_count = 1
    match_mask_color = (255,)* channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def tracbar_canny_threshold_checking(img_):
    """Funkcja, która sprawdza za pomocą tracbarow najlepsze ustawienia Thresoldingu
    Args:
        img_ (array): Obraz
    """    
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('th1', 'Trackbar', 0, 255, nothing)
    cv2.createTrackbar('th2', 'Trackbar',0,255, nothing)
    while True:
        th1 = cv2.getTrackbarPos('th1', 'Trackbar')
        th2 = cv2.getTrackbarPos('th2', 'Trackbar')
        _,thres = cv2.threshold(img_, th1, th2, cv2.THRESH_BINARY)
        canny_image = cv2.Canny(img_, th1, th2)
        cv2.imshow('Thres',thres)
        cv2.imshow('Canny', canny_image)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
def draw_lines(img1, lines):
    """Funkcja, która rysuje linie znalezione na obrazie
    Args:
        img1 (array): Obraz, na którym rysujemy linie
        lines (array): vector lini do narysowania

    Returns:
        array: Obraz z naniesionymi liniami
    """    
    img1 = np.copy(img1)
    blank_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), 3)
    img1 = cv2.addWeighted(img1, 0.8, blank_image, 1, 0.0)
    return img1
def image_detection(image):
    """Wykrywanie linii na obrazie
    Returns:
        array: Obraz na którym narysowalismy linie
    Args:
        image (array): Obraz na którym wykrywamy linie
    """    
    #1.  Define ROI - region zainteresowania 
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    roi_verticies = [
        (0,height),
        (width/2, height/4),
        (width, height)
    ]
    # 2. Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # tracbar_canny_threshold_checking(gray) #sprawdzenie ustawien Thresoldingu
    canny_image = cv2.Canny(gray, 50, 40) # th1 : 50, th2 : 40 - najlepsze ustawienia dla tego obrazu
    cropp_image = region_of_intrest(canny_image, np.array([roi_verticies], np.int32)) # Obraz z ROI (reszta czarna)

    # 3. Color lines
    lines = cv2.HoughLinesP(cropp_image, 3, np.pi/90, 30, lines=np.array([]), minLineLength=20,maxLineGap=25)
    image_with_lines = draw_lines(image, lines)
    plt.subplot(1,3,1) 
    plt.imshow(canny_image)
    plt.title('Canny Image')
    plt.subplot(1,3,2)
    plt.imshow(cropp_image)
    plt.title('Cropp Image')
    plt.subplot(1,3,3)
    plt.imshow(image_with_lines)
    plt.title('With Line Image')
    plt.show()
    return image_with_lines
    
def video_detection(cap):
    """Funkcja, która rysuje linie na video
    Args:
        cap (array): Video
    """    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = image_detection(frame)
        cv2.imshow("Viedo frame",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    
if __name__ == "__main__":
    image = cv2.imread('images/road1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # zeby uzyc matplotlib
    image = image_detection(image)
    
    
    # cap = cv2.VideoCapture('videos/road_movie.mp4')
    # video_detection(cap)
    


