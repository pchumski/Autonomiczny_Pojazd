def initializeTrackbars(intialTracbarVals,wT=320, hT=320):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, empty)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, empty)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, empty)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, empty)