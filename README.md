# Engineering Thesis
## Autonomous Car
### Abstract
Autonomous vehicles have gained a lot of popularity in recent years, making them a hot topic worldwide. This trend is also related to the development and popularization of artificial intelligence. Inspired by this, we created our own project in this field. Using the YOLO (You Only Look Once) algorithm, we developed a system for detecting and recognizing road signs and traffic lights. Additionally, using image processing techniques, we programmed our Nvidia Jetson Jetracer platform to track lines. Moreover, we designed our own controller, which allows remote control of the vehicle. The camera feed can be monitored via a Flask library-based server. The majority of the project is based on Python, while the controller was designed using Kicad software. After assembling and soldering the entire controller circuit, we programmed it using an ESP32 microcontroller. Altogether, our project fits into the concept of autonomous vehicles, which was our goal.
### Platform
![Nvidia Jetson Jetracer](https://github.com/pchumski/Autonomous-Car/blob/main/Lane_detection/picture/pojazd.png)
### Achieved goals
* Remote control of the vehicle - the goal was to develop our own controller that would allow remote control of the platform. We used Bluetooth communication protocol or wireless Wi-Fi network connection.
* Line detection - the goal was to create our own board for the vehicle to move on. Lines were placed on the board to define the path for the platform. Then, using image processing and control algorithm, the vehicle was able to travel along the designated path. The OpenCV and NumPy libraries in Python were used in this project.
* Detection of road signs and traffic lights - the goal was to develop a neural network model for detecting selected road signs and traffic lights in the image. The next step was to create mock-ups of road signs of sizes that would fully cover them in the camera's image. Signs and traffic lights were tested on the previously created line tracking board.
* Website for viewing camera image - the goal was to create a server-side website that would provide remote access to the camera image. This allowed observation of the object detection and line tracking in action.
### Libraries 
* OpenCV
* Numpy
* Flask
* PyTorch
* Yolo
### Links
* [Engineering Thesis](https://github.com/pchumski/Autonomous-Car/blob/main/BSc%20Thesis/BSc_Thesis.pdf)
* [Presentation](https://github.com/pchumski/Autonomous-Car/tree/main/presentation)
* [Picture & Video](https://drive.google.com/drive/u/1/folders/1PUePPLqRdV5ynQXc28LMWLCpgILiKmpQ)
* [Transitional project documentation - old](https://github.com/pchumski/Autonomous-Car/tree/main/transit%20project/documentation)
* [Transitional project video - old](https://drive.google.com/file/d/158aSpdDO3zHkLfEMy2sxm0N-qa4N-M8S/view?usp=sharing)

Last update 02.2023
