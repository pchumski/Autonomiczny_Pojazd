def getLaneCurve(img):
    imgThres = thresholding(img)
    hT, wT, c = img.shape
    points = np.float32([(100, 175), (wT-100, 175),(55, 320 ), (wT-55, 320)])
    imgWarp = warpImg(imgThres, points, wT, hT)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)