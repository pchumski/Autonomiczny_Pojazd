from turtle import width
import pygame

pygame.init() # inicjalizacja okna gry

win = pygame.display.set_mode((500,500)) # wielkosc okna

pygame.display.set_caption("First game") # tytul okna

x = 50
y = 50
width = 40
heigh = 60
vel = 5
# Petla gry
run = True
while run:
    pygame.time.delay(100) # delay
    # Obsluga zdarzen
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        x -= vel
    if keys[pygame.K_RIGHT]:
        x += vel
    if keys[pygame.K_UP]:
        y -= vel
    if keys[pygame.K_DOWN]:
        y += vel
        
    win.fill((0,0,0))
    pygame.draw.rect(win, (255,0,0), (x,y,width,heigh))
    pygame.display.update()
            
pygame.quit()
            