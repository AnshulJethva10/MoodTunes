import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
import webbrowser
import os

# Set up Spotify API credentials
# You need to create an app in Spotify Developer Dashboard to get these
SPOTIFY_CLIENT_ID = 'YOUR_CLIENT_ID'
SPOTIFY_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = "user-read-playback-state,user-modify-playback-state"

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".spotifycache"
))

# Configure playlists/genres for each emotion
emotion_playlists = {
    'Angry': ['spotify:playlist:37i9dQZF1DX1s9knjP51Oa', 'spotify:playlist:37i9dQZF1DX4eRPd9frC1m'],
    'Happy': ['spotify:playlist:37i9dQZF1DXdPec7aLTmlC', 'spotify:playlist:37i9dQZF1DX9XIFQuFvzM4'],
    'Neutral': ['spotify:playlist:37i9dQZF1DX4sWSpwq3LiO', 'spotify:playlist:37i9dQZF1DWZeKCadgRdKQ'],
    'Sad': ['spotify:playlist:37i9dQZF1DX7qK8ma5wgG1', 'spotify:playlist:37i9dQZF1DX889U0CL85jj'],
    'Surprise': ['spotify:playlist:37i9dQZF1DX0BcQWzuB7ZO', 'spotify:playlist:37i9dQZF1DX4fpCWaHpNxy']
}

# Alternative music service options (comment out Spotify and uncomment one of these if preferred)
# For demonstration purposes - these would need their own API implementations
USE_SPOTIFY = True
USE_GAANA = False
USE_JIOSAAVN = False

# Load face detector
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load pre-trained emotion model
classifier = load_model(r'Emotion.h5')

# Define emotion classes
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Variables to track emotion and music state
current_emotion = None
previous_emotion = None
emotion_duration = 0
last_emotion_time = time.time()
is_playing = False
current_playlist_uri = None

def play_emotion_music(emotion):
    """Play music based on detected emotion using Spotify API"""
    global is_playing, current_playlist_uri
    
    if emotion not in emotion_playlists:
        print(f"No playlist defined for emotion: {emotion}")
        return
    
    # Get a random playlist for this emotion
    playlist_uri = random.choice(emotion_playlists[emotion])
    
    if USE_SPOTIFY:
        try:
            # Get available devices
            devices = sp.devices()
            if not devices['devices']:
                print("No active Spotify devices found. Opening web player...")
                webbrowser.open('https://open.spotify.com')
                time.sleep(5)  # Wait for web player to open
                devices = sp.devices()
            
            if devices['devices']:
                device_id = devices['devices'][0]['id']  # Use the first available device
                
                # Start playback from the selected playlist
                if playlist_uri != current_playlist_uri:
                    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
                    print(f"Playing music for {emotion} emotion")
                    is_playing = True
                    current_playlist_uri = playlist_uri
            else:
                print("Still no Spotify devices available")
        except Exception as e:
            print(f"Spotify playback error: {e}")
    
    elif USE_GAANA:
        print(f"Would play Gaana music for {emotion} emotion")
        # Gaana API implementation would go here
    
    elif USE_JIOSAAVN:
        print(f"Would play JioSaavn music for {emotion} emotion")
        # JioSaavn API implementation would go here

def stop_music():
    """Stop currently playing music"""
    global is_playing, current_playlist_uri
    
    if USE_SPOTIFY and is_playing:
        try:
            sp.pause_playback()
            print("Music stopped")
            is_playing = False
            current_playlist_uri = None
        except Exception as e:
            print(f"Error stopping music: {e}")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting emotion detection and music player...")
print("Press 'q' to exit, 's' to skip track, 'p' to pause/play")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # Initialize dominant emotion for this frame
    frame_emotion = None
    highest_confidence = 0
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
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
            emotion_idx = np.argmax(preds[0])
            emotion = class_labels[emotion_idx]
            confidence = np.max(preds[0]) * 100
            
            # If this face has higher confidence than others, use its emotion
            if confidence > highest_confidence:
                highest_confidence = confidence
                frame_emotion = emotion
            
            # Display label and confidence
            label_position = (x, y - 10)
            cv2.putText(frame, f'{emotion} {confidence:.2f}%', 
                        label_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Update emotion tracking
    if frame_emotion:
        current_time = time.time()
        
        # If same emotion as before
        if frame_emotion == current_emotion:
            emotion_duration += current_time - last_emotion_time
        else:
            # New emotion detected
            current_emotion = frame_emotion
            emotion_duration = 0
        
        last_emotion_time = current_time
        
        # Display current emotion and duration
        cv2.putText(frame, f'Sustained: {current_emotion} for {emotion_duration:.1f}s', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Play music if emotion is sustained for 3 seconds and different from previous
        if emotion_duration >= 3 and current_emotion != previous_emotion:
            play_emotion_music(current_emotion)
            previous_emotion = current_emotion
    else:
        # No face detected for this frame
        if time.time() - last_emotion_time > 10:  # If no face for 10 seconds
            stop_music()
            current_emotion = None
            previous_emotion = None
            emotion_duration = 0
    
    # Display control instructions
    cv2.putText(frame, "Press: q=quit, s=skip, p=pause/play", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion-based Music Player', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and USE_SPOTIFY and is_playing:
        try:
            sp.next_track()
            print("Skipped to next track")
        except Exception as e:
            print(f"Error skipping track: {e}")
    elif key == ord('p'):
        if is_playing:
            stop_music()
        elif current_emotion:
            play_emotion_music(current_emotion)

# Clean up
stop_music()
cap.release()
cv2.destroyAllWindows()
print("Application closed")