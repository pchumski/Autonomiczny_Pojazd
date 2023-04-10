middlePoint,imgHist1 = getHistogram(imgWarp, display=True, minPer=0.1, region=2)
curveAveragePoint,imgHist = getHistogram(imgWarp, display=True, minPer=0.9)
curveRaw = curveAveragePoint-middlePoint