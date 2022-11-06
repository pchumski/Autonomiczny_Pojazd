import numpy as np
import cv2
import pandas as pd
import keras

propability_needed = 0.9

labels = pd.read_csv('label_names.csv')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, 640) # zmiena szerokosci
cap.set(4, 480) # zmiana wysokosci
cap.set(10, 180) # zmiana jasnosci

model = keras.models.load_model('traffic_sign.h5')
# print(model.summary())

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

while True:
    _, frame = cap.read()
    
    img = np.asarray(frame)
    img = cv2.resize(img, (32,32))
    img = preprocesing(img)
    cv2.imshow("Preprocesed Image: ", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(frame, "Class: ", (20, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Propability: ", (20, 75), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)
    propabilityValue = np.amax(prediction)
    
    if propabilityValue > propability_needed:
        cv2.putText(frame, str(class_index[0]), (150, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(propabilityValue*100,2))+"%", (150, 75), font, 0.75, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow("result", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()