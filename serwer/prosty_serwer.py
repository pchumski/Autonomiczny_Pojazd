import socket
import os
import cv2

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("192.168.1.12", 30003))
server_socket.listen()
cap = cv2.VideoCapture(0)
i = 0

while True:
    client_socket, client_address = server_socket.accept()
    while True:
        data = client_socket.recv(6)
        #message = int.from_bytes(data, byteorder='big')
        numbers = [b for b in bytearray(data)]
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Received message from {client_address}: {numbers}")
        _,img = cap.read() 
        if numbers[2] == 1:
            cv2.imwrite('screen'+str(i)+'.jpg', img)
            i = i+1
        
    #client_socket.close()
