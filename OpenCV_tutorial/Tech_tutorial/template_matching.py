from turtle import width
import numpy as np
import cv2

#Ladujemy obrazy czarno-biale (zeby latwiej znalezc wzorzec)
img = cv2.imread('assets/soccer_practice.jpg',cv2.IMREAD_GRAYSCALE)
# wzor pilki ktorej szukamy
template = cv2.imread('assets/ball.PNG', cv2.IMREAD_GRAYSCALE)
# wzor buta ktory szukamy
template1 = cv2.imread('assets/shoe.PNG', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (0,0), fx=0.75,fy=0.75)
template = cv2.resize(template, (0,0), fx=0.75,fy=0.75)
template1 = cv2.resize(template1, (0,0), fx=0.75,fy=0.75)


#Bierzemy wysokosc i szerokosc naszego obrazka wzorcowego ktory chcemy znalezc (poniewaz jest czarno bialy jego shape to: (height, width))
#Kolorowy mialby shape: (height, width, channels)
h, w = template.shape 

h1, w1 = template1.shape 

# Metody do dopasowywania wzorce (Wiecej w dokumentacji)
methods = [ cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# Sprawdzimy wszystkie metody i zobaczymy, ktora bedzie najlepsza
for method in methods:
    img2 = img.copy() # kopiuje oryginalny obraz zeby zostawic go do porownania 
    
    result = cv2.matchTemplate(img2, template, method)
    #Ta funkcja zwroci nam tablice gdzie bedziemy miec pokazane, ktory element obrazu jest najlepiej dopasowany 
    #Bedzie to obraz o wielkosci (W-w+1, H-h+1), gdzie H,W to wielkosci obrazu bazowego, a w,h to wielkosci wzoru (template)
    
    result1 = cv2.matchTemplate(img2, template1, method)
    
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # Funkcja zwraca nam maksymalne i minimalne wartosci dopasowania oraz ich lokacje na obrazie bazowym
    # min_loc i max_loc to wartosci polozenia gornego lewgo rogu, gdzie nasz wzor jest dopasowany 
    
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1) 
    
    # Dla dwoch metod najlepsza bedzie wartosc minimalna, a dla reszty maksymalna 
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
        location1 = min_loc1
    else:
        location = max_loc
        location1 = max_loc1
        
    bootom_right = (location[0] + w, location[1] + h) # zeby narysowac prosotkat potrzebujemy tez prawego dolnego rogu naszego miejsca gdzie jest znaleziony wzor
    bootom_right1 = (location1[0] + w1, location1[1] + h1)
    cv2.rectangle(img2, location, bootom_right, 255, 5)
    
    cv2.rectangle(img2, location1, bootom_right1, 0, 5)
    
    cv2.imshow('Match',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

