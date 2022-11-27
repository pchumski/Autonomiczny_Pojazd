import torch
import cv2
import numpy as np
import onnxruntime

img = cv2.imread('road561.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,640))

print(torch.cuda.is_available())
font = cv2.FONT_HERSHEY_SIMPLEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yolov5','custom', 'best3.pt',source='local', force_reload=True)
model.cuda().eval().half()
model.conf = 0.5
model.iou = 0.45
model.multi_label = False
model.max_det = 10



cap = cv2.VideoCapture(0)
cap.set(3, 320) # zmiena szerokosci
cap.set(4, 320) # zmiana wysokosci



while True:
    ret,frame = cap.read()
    
    framec = np.copy(frame)
    framec = cv2.cvtColor(framec, cv2.COLOR_BGR2RGB)

    results = model(framec, size=320)

    #results.print()

    #print(results.pandas().xyxy)

    for i in range(len(results.pandas().xyxy[0])):
        wynik = results.pandas().xyxy[0]['confidence'][i]
        if float(wynik) >= 0.5:
            xmin = results.pandas().xyxy[0]['xmin'][i]
            ymin = results.pandas().xyxy[0]['ymin'][i]
            xmax = results.pandas().xyxy[0]['xmax'][i]
            ymax = results.pandas().xyxy[0]['ymax'][i]
            klasa = results.pandas().xyxy[0]['name'][i]
            object_width = round((xmax - xmin),1)
            
            cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
            cv2.putText(frame, str(klasa), (int(xmin), int(ymin)-35), font, 1, (0,0,0), 1)
            cv2.putText(frame, str(round(wynik,2)), (int(xmin), int(ymin)-15), font, 1, (0,0,0), 1)
            
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#cv2.imshow("Result", img)
#cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()



