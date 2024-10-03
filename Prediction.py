import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.keras')

# Define class labels in the order they were during training
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face ROI and resize it to the target size required by the model (48x48 pixels)
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (48, 48))
        img_pixels = np.expand_dims(roi_color, axis=0)

        # Normalize the image data
        img_pixels = img_pixels.astype('float32') / 255

        # Make predictions using the loaded model
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = class_labels[max_index]

        # Display the predicted emotion
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    
    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
