def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0, 0, 80])
    upperWhite = np.array([179, 50, 180])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite