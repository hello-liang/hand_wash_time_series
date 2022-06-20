# -*- coding: utf-8 -*-
"""
for the mutiprocess code , the video will complete very quickly ,so have to use camera
Created on Wed Nov 10 10:40:52 2021
only change line 39 41 42 and 126 ,134,and11~13 change the label number and path of guide   and guide image size
@author: base tensorflow's lite'
"""
# ok ,model load must put in the second processer function
import os
#os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

num_frame_analysis=30
import numpy
import numpy as np
import re
import cv2
import time
import mediapipe as mp
from multiprocessing import Process
from multiprocessing import Queue
from skel_aug import skele_augmentation
from utils import process_output_skelenton_to_array

import pickle
f= open('store.pckl','rb')
model_params=pickle.load(f)
f.close()


f = open('classifier.pckl', 'rb')
classifier = pickle.load(f)
f.close()

def classify_frame(process_output_skelenton_to_array,skele_augmentation,classifier,model_params, inputQueue,outputQueue):  # only for input a frame ,output a accuracy,that's all



    # keep looping#need loop all of the time !!!!
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:

            # check to see if there is a frame in our input queue
            if not inputQueue.empty():


                test_frames = inputQueue.get()
                start = time.time()
                skeleton_data = []
                for i in range(len(test_frames)):
                    skeleton = hands.process(test_frames[i])
                    skeleton_data.append(process_output_skelenton_to_array(skeleton))
                # here the input is a two frames .but only use the first one

                data = np.float64(np.array(skeleton_data))
                data_AUG = np.float64(skele_augmentation(data, model_params))
                data_AUG = numpy.expand_dims(data_AUG, axis=0)
                prediction = classifier.predict(data_AUG)
                detections = prediction

                # write the detections to the output queue
                outputQueue.put(detections)


# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

if __name__ == '__main__':

    # initialize the input queue (frames), output queue (detections),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    detections = None
    # construct a child process *indepedent* from our main process of

    p = Process(target=classify_frame, args=( process_output_skelenton_to_array,skele_augmentation,classifier, model_params, inputQueue,
                                                 outputQueue,))
    p.daemon = True
    p.start()

    result_max="begin"
        #    cap = cv2.VideoCapture("/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/HandWashDataset_self/Step6/Step6_24.avi")
    cap = cv2.VideoCapture(0)  # 2
    test_frames = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(test_frames) < num_frame_analysis:  # maybe need over the tcn length?
            test_frames.append(image)
        else:

            if inputQueue.empty():
                print(111)
                inputQueue.put(test_frames)
                print(222)

            # if the output queue *is not* empty, grab the detections
            if not outputQueue.empty():

                detections = outputQueue.get()
            # draw the detections on the frame)
            if detections is not None:
                print(333)

                prediction = detections
                result_max = str(prediction[0]+1)

            test_frames = []

        cv2.putText(image, result_max, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 0)  # (col,row) begin
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

