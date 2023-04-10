def getHistogram(img, minPer=0.1, display=False, region=1):

    if region ==1:
        histValues = np.sum(img, axis=0)
    else :
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
    #histValues = np.sum(img, axis=0)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    indexArray =np.where(histValues >= minValue)
    basePoint =  int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(int(x),img.shape[0]),(int(x),img.shape[0]-int(intensity//255//region)),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint