import pygame # biblioteka do obsługi pada
from jetracer.nvidia_racecar import NvidiaRacecar # Klasa samochodzika

      
if __name__ == "__main__":
    # Inicjalizacja
    pygame.init()
    j = pygame.joystick.Joystick(0)
    j.init()
    print('Joystick initialized: %s' % j.get_name() )
    clock = pygame.time.Clock()
    car = NvidiaRacecar()
    car.throttle_gain = 0.5
    car.steering_offset = -0.18
    # Petla programu
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if j.get_button(11):
                break
        x_speed = round(j.get_axis(2),2)
        y_speed = round(j.get_axis(1),2)
        print(f'x_speed: {x_speed},  y_speed: {y_speed}')
        car.throttle = y_speed
        car.steering = x_speed
        clock.tick(180)
        
