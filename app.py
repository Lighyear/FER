import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
@st.cache_resource
def load_emotion_model():
    return load_model('fer_ck_cnn_improved_model.h5')

model = load_emotion_model()

# Emotion labels map
emotion_map = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# App title
st.title("😄 Facial Emotion Recognition (Web-Based)")

# Camera input
img_data = st.camera_input("Take a picture")

if img_data is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please try again.")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            with st.spinner("Analyzing emotion..."):
                preds = model.predict(roi, verbose=0)[0]
                label = emotion_map[np.argmax(preds)]
                confidence = np.max(preds)

            # Annotate prediction on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert BGR to RGB for display in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Detected Emotions", use_column_width=True)
