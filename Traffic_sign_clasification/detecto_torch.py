import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import time

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights='FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT'
    )
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
# Sprawdzenie czy GPU jest aktywne
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()



#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
labels = ['background','speedlimit', 'stop', 'crosswalk', 'trafficlight']

model = create_model(num_classes=5)
checkpoint = torch.load('model_weights_sign.pth', map_location=device)
# print(checkpoint)
model.load_state_dict(checkpoint)
model.to(device).eval()

cap = cv2.VideoCapture(0)
detection_threshold = 0.9
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, orig_frame = cap.read()
    
    frame = orig_frame.copy()
    frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    # frame = cv2.resize(frame, (300,300))
    frame /= 255.0
    frame = np.transpose(frame, (2,0,1)).astype(np.float32)
    frame = torch.tensor(frame, dtype=torch.float).cuda()
    frame = torch.unsqueeze(frame, 0)
    start_time = time.time()
    with torch.no_grad():
        # get predictions for the current frame
        outputs = model(frame.to(device))
    end_time = time.time()
    # print(outputs)
    scores = outputs[0]['scores'].tolist()
    classes = outputs[0]['labels'].tolist()
    boxes = outputs[0]['boxes'].tolist()
    if (max(scores) >= detection_threshold):
        for i in range(len(scores)):
            if scores[i] >= detection_threshold:
                napis = labels[classes[i]]
                x1,y1,x2,y2 = boxes[i]
                # print(x1,y1,x2,y2)
                # print(type(x1))
                cv2.putText(orig_frame, napis, (int(x1), int(y1-10)), font, 0.7, (255,0,0), 2)
                cv2.rectangle(orig_frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1)
    # print(scores)
    # outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # scores = [float(i) for i in outputs[0]['scores'].detach().numpy()]
    # if len(scores) > 0:
    #     boxes = [i for i in outputs[0]['boxes'].detach().numpy()]
    #     # print(boxes)
    #     score = max(scores)
    #     boxes = boxes[scores == score].astype(np.int32)
    #     d_boxes = boxes.copy()
    #     pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
    #     lab = pred_classes[scores == score]
    #     if score > detection_threshold:
    #         for j, box in enumerate(d_boxes):
    #             class_name = pred_classes[j]
    #             color = (0,255,0)
    #             orig_image = draw_boxes(orig_image, box, color, 1)
    cv2.imshow("Frame", orig_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# model = Model(labels, model_name=Model.MOBILENET_320)
# model.get_internal_model().load_state_dict(torch.load('model_weights_sign.pth', map_location=model._device))

# visualize.detect_live(model, 0.85)