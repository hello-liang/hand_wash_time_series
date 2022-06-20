# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:19:18 2021

@author: asabater
"""
#hyper parameter
num_frame_analysis=30
import numpy
import numpy as np
from skel_aug import skele_augmentation
import pickle

from utils import process_output_skelenton_to_array
import re
f= open('store.pckl','rb')
model_params=pickle.load(f)
f.close()


f = open('classifier_30fps.pckl', 'rb')
classifier = pickle.load(f)
f.close()
#model = keras.models.load_model("my_h5_model.h5")

import time


np.random.seed(0)

# for this file ,get the skeleton of kaggle dataset and try test
import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import re



test_frames=[]
# For static images:
result_max = "begin"
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

#    cap = cv2.VideoCapture("/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/HandWashDataset_self/Step6/Step6_24.avi")
    cap = cv2.VideoCapture(0)#2
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(test_frames)<num_frame_analysis:# maybe need over the tcn length?
            test_frames.append(image)
        else:
            start = time.time()
            skeleton_data=[]
            for i in range(len(test_frames)):
                skeleton=hands.process(test_frames[i])
                skeleton_data.append(process_output_skelenton_to_array(skeleton))
            # here the input is a two frames .but only use the first one
            data= np.float64(np.array(skeleton_data))
            data_AUG = np.float64(skele_augmentation(data, model_params))
            data_AUG=numpy.expand_dims(data_AUG ,axis=0)
            prediction = classifier.predict(data_AUG)
            test_frames=[]
            result_max=str(prediction[0]+1)
            end = time.time()
            print(end - start)
        cv2.putText(image,result_max ,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0), 2, 0)   #(col,row) begin
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()


