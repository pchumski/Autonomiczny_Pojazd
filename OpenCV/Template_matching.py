import cv2
import numpy as np

img = cv2.imread('assets/soccer_practice.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('assets/ball.PNG', 0)
w,h = template.shape[::-1] # wydobywamy szerokosc i wysokosc (jest to czarno bialy obraz wiec jest tylko jeden kanal)

res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# szukamy najjasniejszego punktu res,czyli tego ktory wedlug funkcji pasuje najlepiej do naszego wzoru
threshold = 0.925 # dopasowujemy zeby znalezc najbardziej prawdopodone rogi polozenia naszego wzoru
loc = np.where(res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):
    print(pt)
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255,0,0), 3)
    


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()