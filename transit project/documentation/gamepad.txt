#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32

#Initialize pygame and gamepad
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
print ('Initialized Joystick : %s' % j.get_name())

def teleop_gamepad():
    #Setup topics publishing and nodes
    pub_throttle = rospy.Publisher('throttle', Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering', Float32, queue_size=8)
    rospy.init_node('teleop_gamepad', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        pygame.event.pump()
        
        #Obtain gamepad values
        throttle = j.get_axis(1) #Left thumbstick Y
        steering = j.get_axis(2) #Right thumbstick X
        print("Throttle:", throttle)
        print("Steering:", steering)

        #Pubblish gamepad values
        pub_throttle.publish(throttle)
        pub_steering.publish(steering)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        teleop_gamepad()
    except rospy.ROSInterruptException:
        pass
