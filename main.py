import cv2
import os
import tensorflow as tf
import numpy as np
from pygame import mixer
import time

# Initialize the mixer module for playing sound
mixer.init()
sound = mixer.Sound('alarm.wav')  # Load the alarm sound file

# Load pre-trained Haar cascade classifiers for face and eyes detection
face = cv2.CascadeClassifier(
    'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(
    'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    'haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

# Load the trained model for eye state prediction
try:
    model = tf.keras.models.load_model('models/cnnCat2.h5')
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Initialize webcam
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# Main loop to continuously capture frames from the webcam
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes using the classifiers
    faces = face.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle at the bottom of the frame to display status messages
    cv2.rectangle(frame, (0, height-50), (200, height),
                  (0, 0, 0), thickness=cv2.FILLED)

    # Detect and annotate face on the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Process right eye data for prediction
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        lbl = 'Closed' if (rpred[0][0] > rpred[0][1]) else 'Open'
        break

    # Process left eye data for prediction
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        lbl = 'Closed' if (lpred[0][0] > lpred[0][1]) else 'Open'
        break

    # Update score based on eye predictions and display the eye status
    if (rpred[0][0] > rpred[0][1] and lpred[0][0] > lpred[0][1]):
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    # Prevent score from going negative
    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # If the score is high, indicating potential drowsiness, play an alarm
    if score > 15:
        # Save a frame as an image file
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()  # Play the alarm sound
        except:
            pass
        thicc = min(16, thicc + 2) if thicc < 16 else max(2, thicc - 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255),
                      thicc)  # Flash a red frame border

    # Display the processed frame and check for quit key
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()
