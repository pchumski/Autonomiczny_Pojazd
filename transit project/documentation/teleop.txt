#!/usr/bin/env python3

import rospy
from getkey import getkey
from std_msgs.msg import Float32

def teleop():
    #Setup topics publishing and nodes
    pub_throttle = rospy.Publisher('throttle', Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering', Float32, queue_size=8)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    #Print contol hints
    print("Running teleop.py")
    print("W: Forward")
    print("A: Steer Left")
    print("S: Backward")
    print("D: Steer Right")
    print("Q: Throttle stop")
    print("W: Steering reset")

    while not rospy.is_shutdown():
        #Get key press
        key = getkey()

        #Throttle
        if key == 'w':
            pub_throttle.publish(-1.0)
        elif key == 's':
            pub_throttle.publish(1.0)
        elif key == 'q':
            pub_throttle.publish(0.0)

        #Steering
        if key == 'a':
            pub_steering.publish(-1.0)
        elif key == 'd':
            pub_steering.publish(1.0)
        elif key == 'e':
            pub_steering.publish(0.0)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        teleop()
    except rospy.ROSInterruptException:
        pass
