import numpy as np
import cv2
import pandas as pd
#import keras

propability_needed = 0.9

#labels = pd.read_csv('label_names.csv')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, 640) # zmiena szerokosci
cap.set(4, 480) # zmiana wysokosci
cap.set(10, 180) # zmiana jasnosci

#model = keras.models.load_model('traffic_sign.h5')
# print(model.summary())
def nothing(x):
    pass
def returnredness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocesing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

# print(labels['SignName'][8])
cv2.namedWindow("Thres")
cv2.createTrackbar("th1", "Thres", 0, 255,nothing)
cv2.createTrackbar("th2", "Thres", 0, 255,nothing)

while True:
    _, frame = cap.read()
    
    redness = returnredness(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3), 0.1)
    th1 = cv2.getTrackbarPos("th1", "Thres")
    th2 = cv2.getTrackbarPos("th2", "Thres")
    _, img = cv2.threshold(redness, th1, th2, cv2.THRESH_BINARY)
    
    canny = cv2.Canny(blur, th1, th2)
    try:
        countours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                # cv2.drawContours(imgCountour, cnt, -1, (255,0,255), 7)
                # peri = cv2.arcLength(cnt, True)
                # approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                x_, y_, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(frame, cnt, -1, (255,0,255), 7)
                cv2.rectangle(frame, (x_, y_), (x_+w, y_+h), (0,255,0), 5)
        cv2.imshow("frame", frame)
    except:
        cv2.imshow("frame", frame)
    #cv2.imshow("Threshold", img)
    #cv2.imshow("Canny", canny)
    # img = np.asarray(frame)
    # img = cv2.resize(img, (32,32))
    # img = preprocesing(img)
    # cv2.imshow("Preprocesed Image: ", img)
    # img = img.reshape(1, 32, 32, 1)
    # cv2.putText(frame, "Class: ", (20, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    # cv2.putText(frame, "Propability: ", (20, 75), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    
    # prediction = model.predict(img)
    # class_index = np.argmax(prediction, axis=1)
    # propabilityValue = np.amax(prediction)
    
    # if propabilityValue > propability_needed:
    #     cv2.putText(frame, str(class_index[0]), (150, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    #     cv2.putText(frame, str(round(propabilityValue*100,2))+"%", (150, 75), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    #cv2.imshow("result", img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()