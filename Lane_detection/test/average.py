curveList.append(curveRaw)
if len(curveList) > avgVal:
    curveList.pop(0)
curve = int(sum(curveList)/len(curveList))