#  like others this is for data set

import cv2
import mediapipe as mp
import numpy as np
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import re
print("recent work path should be ..../hand_wash_time_series")
print(os.getcwd()) #should be (...../hand_wash_time_series)

input="archive"+os.sep+"HandWashDataset"+os.sep+"HandWashDataset"

output="skeleton_hand_wash_kaggle_one"
print(os.getcwd())  # should be (/media/liang/ssd2/wash_hand_3/hand_wash_time_series)


# 50 guys do 6 action
def process_output_skelenton_to_array_muti_hand(results):
    # not sure the type of mediapipe output ,I use this function convert it to array
    out = np.ones(126)
    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        out = out
        # can not find a hand ,initialize to 0
    elif len(results.multi_handedness) == 1:

        results_class = str(results.multi_handedness[0])
        results_class = re.split('label: "|"\n}\n', results_class)
        results_class = results_class[1]

        if results_class == "Left":
            hand_landmarks = str(results.multi_hand_landmarks[0])
            hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
            out[0:63] = hand_landmarks[1:64]
        else:
            hand_landmarks = str(results.multi_hand_landmarks[0])
            hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
            out[63:126] = hand_landmarks[1:64]

    elif len(results.multi_handedness) == 2:  # 2 hand right first then left

        hand_landmarks = results.multi_hand_landmarks[0]
        hand_landmarks = str(hand_landmarks)
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out[0:63] = hand_landmarks[1:64]
        hand_landmarks = results.multi_hand_landmarks[1]
        hand_landmarks = str(hand_landmarks)
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out[63:126] = hand_landmarks[1:64]
    else:
        print("have more than two hand？")
        print(len(results.multi_handedness))
    return str(out.tolist())
def process_output_skelenton_to_array(results):
    # not sure the type of mediapipe output ,I use this function convert it to array
    out = ['1'] * 63
    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        out = out
        # can not find a hand ,initialize to 0
    else:
        # only choose the first one hand
        hand_landmarks = str(results.multi_hand_landmarks[0])
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out = hand_landmarks[1:64]
    return out


# For static images:
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

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

