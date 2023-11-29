import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Open the camera
captureDevice = cv2.VideoCapture(0)

# Detector of Hand
detector = HandDetector(maxHands=1)

paddingOffset = 40


while True:
    # Read a frame from the camera
    success, frame = captureDevice.read()
    
    # Find hands in the frame
    hands, frame = detector.findHands(frame, draw=False)

    if hands : 
        selectedHand = hands[0]
        x,y,w,h = selectedHand['bbox']

        # frameWhiteBackground

        frameCrop = frame[y-paddingOffset : y+h+paddingOffset , x-paddingOffset : x+w +paddingOffset]
        cv2.imshow("Hand", frameCrop)


    # Display the frame with hand detection
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
captureDevice.release()
cv2.destroyAllWindows()
