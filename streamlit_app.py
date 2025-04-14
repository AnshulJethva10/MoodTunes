import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import streamlit.components.v1 as components

# Load model and classifier
model = load_model('Emotion.h5', compile=False)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion -> YouTube Playlist (embeddable links)
emotion_links = {
    'Angry': 'https://www.youtube.com/embed/videoseries?list=PLp_LwqrLksxeynGj6LyX9r89NpF9_bNCF&autoplay=1',
    'Happy': 'https://www.youtube.com/embed/videoseries?list=PLGb4vbMWyI-10b064S09MgvspGFOGQpBo&autoplay=1',
    'Neutral': 'https://www.youtube.com/embed/videoseries?list=PLzKILxYC79RDspOFfHUselpBLOVEOJO_T&autoplay=1',
    'Sad': 'https://www.youtube.com/embed/videoseries?list=PL9khxBZiiQwoKEqdTrb4ip-S_Tov6FkBQ&autoplay=1',
    'Surprise': 'https://www.youtube.com/embed/videoseries?list=PLO7-VO1D0_6M1xUjj8HxTxskouWx48SNw&autoplay=1'
}

# Session state init
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'emotion' not in st.session_state:
    st.session_state.emotion = None
if 'fixed_emotion' not in st.session_state:
    st.session_state.fixed_emotion = None

# Streamlit UI
st.title("🎭 Emotion Detector with Playlist")
st.markdown("Detect your emotion using webcam after a countdown, or upload a photo, and get a matching YouTube playlist 🎵")

# Step 1: Capture or Upload Image
use_webcam = False
if st.button("📸 Start Detection"):
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        st.warning("⚠️ Unable to access webcam. Please upload an image instead.")
    else:
        use_webcam = True
        st.markdown("### ⏱️ Countdown:")
        countdown_text = st.empty()

        for i in range(3, 0, -1):
            countdown_text.markdown(f"**{i}...**")
            time.sleep(1)
        countdown_text.markdown("**Capturing Image...**")
        time.sleep(0.5)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to capture image from webcam.")
            use_webcam = False
        else:
            st.session_state.frame = frame

# Fallback: Upload Image
if not use_webcam and st.session_state.frame is None:
    uploaded_file = st.file_uploader("📤 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.session_state.frame = frame

# Step 2: Detect Emotion
if st.session_state.frame is not None and st.session_state.emotion is None:
    frame = st.session_state.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    emotion = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            emotion = class_labels[np.argmax(preds)]
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    st.session_state.frame = frame
    st.session_state.emotion = emotion
    st.session_state.fixed_emotion = None

# Step 3: Show Detection Results
if st.session_state.frame is not None:
    st.image(cv2.cvtColor(st.session_state.frame, cv2.COLOR_BGR2RGB), caption="Detected Frame")
    if st.session_state.emotion:
        st.success(f"Detected Emotion: **{st.session_state.emotion}**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Fix Emotion"):
                st.session_state.fixed_emotion = st.session_state.emotion
        with col2:
            if st.button("🔁 Redetect"):
                st.session_state.frame = None
                st.session_state.emotion = None
                st.session_state.fixed_emotion = None
                st.experimental_rerun()
    else:
        st.warning("No face detected. Please try again.")

# Step 4: Show Playlist
if st.session_state.fixed_emotion:
    st.markdown(f"### 🎶 Playing {st.session_state.fixed_emotion} Playlist:")
    components.iframe(emotion_links[st.session_state.fixed_emotion], height=400)
