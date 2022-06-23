#  like others this is for data set

import cv2
import mediapipe as mp
import numpy as np
import os

from utils import process_output_skelenton_to_array

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import re
print("recent work path should be ..../hand_wash_time_series")
print(os.getcwd()) #should be (...../hand_wash_time_series)

#input="archive"+os.sep+"HandWashDataset"+os.sep+"HandWashDataset"
input="magic_mirror_data"

output="skeleton_magic_mirror_2_hand"
print(os.getcwd())  # should be (/media/liang/ssd2/wash_hand_3/hand_wash_time_series)



# For static images:
with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    for action in os.listdir( input+os.sep):
        v_i = 0
        for video in os.listdir( input+os.sep + action):
            v_i = v_i + 1
            # For webcam input:
            cap = cv2.VideoCapture( input+os.sep + action + os.sep + video)
            video_name = video.split('.')[0]
            subject = video_name[-1]
            path_joint =  output+os.sep + str(v_i) + os.sep + action + os.sep
            if not os.path.exists(path_joint):
                os.makedirs(path_joint)
            f = open(path_joint + 'joint.txt', 'w')
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                list_skeleton = process_output_skelenton_to_array(results)
                skeleton_array = ' '.join(list_skeleton)
                f.write(skeleton_array + '\n')
            f.close()
            cap.release()

