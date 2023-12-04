import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Load your pre-trained EfficientNetB0 model
model = load_model("path/to/your/model.h5")  # Replace with the actual path to your model

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

    if hands:
        selectedHand = hands[0]
        x, y, w, h = selectedHand['bbox']

        # Crop the hand region
        frameCrop = frame[y - paddingOffset: y + h + paddingOffset, x - paddingOffset: x + w + paddingOffset]

        # Preprocess the cropped hand image for EfficientNetB0
        processed_hand = preprocess_input(cv2.resize(frameCrop, (224, 224)))

        # Expand dimensions to match the input shape expected by the model
        processed_hand = np.expand_dims(processed_hand, axis=0)

        # Now 'processed_hand' is ready to be passed to the EfficientNetB0 model for prediction
        prediction = model.predict(processed_hand)

        # You can use 'prediction' for further processing or display
        print("Model Prediction:", prediction)

        cv2.imshow("Hand", frameCrop)

    # Display the frame with hand detection
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
captureDevice.release()
cv2.destroyAllWindows()
