def gstreamer_pipeline(
    # The ID of the camera sensor to use
    sensor_id=0, 
    # The width of the frames captured from the camera
    capture_width=320, 
    # The height of the frames captured from the camera
    capture_height=320,
    # The width of the frames displayed on the screen
    display_width=320, 
    # The height of the frames displayed on the screen
    display_height=320, 
    # The number of frames per second captured from the camera
    framerate=30, 
    # The method used to flip the frames captured from the camera (0 = no flipping)
    flip_method=0, 
):
    return (
        # The element that captures video from the camera sensor, using the sensor ID specified in the input
        "nvarguscamerasrc sensor-id=%d !" 
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        # The element that performs video format conversion, it also uses the flip_method to flip the frames
        "nvvidconv flip-method=%d ! " 
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        # The element that performs color space conversion, the format is BGRx, then it is converted to BGR
        "videoconvert ! " 
        # The element that receives the final processed frames.
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