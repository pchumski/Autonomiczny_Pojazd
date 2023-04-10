# Function calculates the histogram of an image and returns the base point of the histogram, with an option to display the histogram
def getHistogram(img, minPer=0.1, display=False):
    histValues = np.sum(img, axis=0)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    indexArray =np.where(histValues >= minValue)
    basePoint =  int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint