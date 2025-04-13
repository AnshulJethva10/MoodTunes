import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import time

# Load face detector
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load pre-trained model
classifier = load_model(r'Emotion.h5')

# Define emotion classes
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    labels = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]  # Fixed bug: was using x:x+h
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            # Preprocess the image
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Make prediction using TensorFlow
            with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
                preds = classifier.predict(roi, verbose=0)
            
            # Get emotion with highest probability
            label = class_labels[np.argmax(preds[0])]
            
            # Calculate confidence
            confidence = np.max(preds[0]) * 100
            
            # Display label and confidence
            label_position = (x, y - 10)
            cv2.putText(frame, f'{label} {confidence:.2f}%', 
                        label_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()