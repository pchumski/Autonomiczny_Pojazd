def getLaneCurve(img):
    imgCopy = img.copy()
    imgThres = thresholding(img)
    hT, wT, c = img.shape
    points = valTrackbars()
    imgWarp = warpImg(imgThres, points, wT, hT)
    imgWarpPoints = drawPoints(imgCopy, points)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('WarpPoints', imgWarpPoints)