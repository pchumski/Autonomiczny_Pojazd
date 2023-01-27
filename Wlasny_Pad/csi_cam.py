from jetcam.csi_camera import CSICamera
import cv2

camera = CSICamera(width=320, height=320, capture_width=1080, capture_height=720, capture_fps=60)



while True:
    
    image = camera.read()

    cv2.imshow("obraz", image)

    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()