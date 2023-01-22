import os
import cv2
import numpy as np
import keras
import datetime
# from tensorflow.keras.utils import img_to_array as image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import pandas as pd
from analysedata import analysedata
# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
#defining stas variables
prev_emotion="happy"
emotion_timestamps = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
prevTime=time.time()
while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)


    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        # print("Predicted values",predictions)
        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        if prev_emotion != predicted_emotion:
            currTime=time.time()
            timeLast=currTime - prevTime
            emotion_timestamps[prev_emotion] = emotion_timestamps[prev_emotion] + timeLast
            prev_emotion=predicted_emotion
            prevTime=currTime
        #     emotion_timestamps[predicted_emotion] = time.time()
        #     prev_emotion = predicted_emotion
        # elif prev_emotion == predicted_emotion:
        #     current_time = time.time()
        #     last_changed_time = emotion_timestamps[predicted_emotion]
        #     total_seconds = current_time - last_changed_time
        #     emotion_timestamps[predicted_emotion] = total_seconds

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1200, 720))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows


pathval='./userdata.csv'

if(os.path.exists(pathval)):
    dataval = pd.DataFrame(emotion_timestamps, index=[0])
    datasetval = pd.read_csv('./userdata.csv')
    analysedata(datasetval, dataval, pathval)
    datasetval = datasetval.append(emotion_timestamps, ignore_index=True)
    datasetval.to_csv('userdata.csv')

else:
    dataval = pd.DataFrame(emotion_timestamps, index=[0])
    print("Oops cannot show your stats, as your data is first dataset in our file\n Although if you want timestamps then here we go\n",dataval)

    dataval.to_csv('userdata.csv')

