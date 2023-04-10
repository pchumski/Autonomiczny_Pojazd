import socket
import os
import cv2
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar() # Tworzenie obiektu car z biblioteki Nvidii do sterowania pojazdem
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=320,
    capture_height=320,
    display_width=320,
    display_height=320,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("192.168.1.13", 30003))
server_socket.listen()
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
i = 0
throotle = 0
stering = 0
pot = 0

car.throttle = 0.0
car.throttle_gain = 0.3

car.steering_gain= 0.4
car.steering_offset = -0.2
car.steering = 0.0


while True:
    client_socket, client_address = server_socket.accept()
    while True:
        data = client_socket.recv(6)
        #message = int.from_bytes(data, byteorder='big')
        numbers = [b for b in bytearray(data)]
        # os.system('cls' if os.name == 'nt' else 'clear')
        #print(f"Received message from {client_address}: {numbers}")
        _,img = cap.read()
        # Robienie zrzutu kamery 
        if numbers[2] == 1:
            cv2.imwrite('zd_serwer/screen'+str(i)+'.jpg', img)
            i = i+1
        if numbers[4] == 1:
            break
            #break
        #stering = (float(numbers[0]) - 127.5)/127.5 # normalizacja
        #throotle = (float(numbers[1]) - 127.5)/127.5 # normalizacja
        car.throttle = -(float(numbers[1]) - 127.5)/127.5
        car.steering = -(float(numbers[0]) - 127.5)/127.5 
        #throotle = (float(numbers[1]) - 127.5)/127.5 # normalizacja
        pot = float(numbers[3])/255
        if abs(car.steering) <0.1:
            car.steering = 0.0
        if abs(car.throttle) <0.1:
            car.throttle = 0.0
        #os.system('cls' if os.name == 'nt' else 'clear')
        #print(f'Receive from {client_address}: Throtle: {car.throttle}, Stering: {car.steering}, Push: {numbers[2]}, Value: {pot}')
        #cv2.imshow("Frame", img)
        #if cv2.waitKey(10) == ord('q'):
            #break
    cap.release()
    # closing all opend windows
    cv2.destroyAllWindows()
    client_socket.close()
    break
        
        
        
        
    #client_socket.close()
