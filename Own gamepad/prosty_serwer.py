import socket
import os
import cv2
import sys
#from jetracer.nvidia_racecar import NvidiaRacecar

#car = NvidiaRacecar() # Tworzenie obiektu car z biblioteki Nvidii do sterowania pojazdem

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("192.168.1.12", 30003))
server_socket.listen()
cap = cv2.VideoCapture(0)
i = 0
throotle = 0
stering = 0
pot = 0

# car.throttle = 0.0
# car.throttle_gain = 0.3

# car.steering_gain= 0.4
# car.steering_offset = -0.2
# car.steering = 0.0


while True:
    client_socket, client_address = server_socket.accept()
    while True:
        data = client_socket.recv(8)
        #message = int.from_bytes(data, byteorder='big')
        numbers = [b for b in bytearray(data)]
        #os.system('cls' if os.name == 'nt' else 'clear')
        #print(f"Received message from {client_address}: {numbers}")
        _,img = cap.read()
        # Robienie zrzutu kamery 
        if numbers[2] == 1:
            cv2.imwrite('screen'+str(i)+'.jpg', img)
            #print("Zdjecie")
            i = i+1
        if numbers[7] == 1:
            sys.exit(0)
        stering = (float(numbers[0]) - 127.5)/127.5 # normalizacja
        throotle = (float(numbers[1]) - 127.5)/127.5 # normalizacja
        # car.throttle = (float(numbers[1]) - 127.5)/127.5
        # car.steering = (float(numbers[0]) - 127.5)/127.5 
        #throotle = (float(numbers[1]) - 127.5)/127.5 # normalizacja
        pot = float(numbers[3])/255
        if abs(stering) <0.1:
            stering = 0.0
        if abs(throotle) <0.1:
            throttle = 0.0
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Receive from {client_address}: Throtle: {throotle}, Stering: {stering}, Push: {numbers[2]}, Value: {pot}')
        
        
        
        
    #client_socket.close()
