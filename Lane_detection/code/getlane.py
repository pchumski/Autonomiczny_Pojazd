def getLaneCurve(img):
    imgThres = thresholding(img)
    cv2.imshow('tresh', imgThres)