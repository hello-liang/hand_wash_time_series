# -*- coding: utf-8 -*-
"""
for the mutiprocess code , the video will complete very quickly ,so have to use camera
Created on Wed Nov 10 10:40:52 2021
only change line 39 41 42 and 126 ,134,and11~13 change the label number and path of guide   and guide image size
@author: base tensorflow's lite'
"""
# ok ,model load must put in the second processer function
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import numpy as np
import cv2
import time
import sys
from data_generator import DataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from collections import deque
import mediapipe as mp
# import the necessary packages
from multiprocessing import Process
from multiprocessing import Queue
import random
import prediction_utils
import time


thr=0.1
time_gap=(60)/6 #how many time for wash hand each step
input_length = 126
recent_color=225
root="/home/liang/Desktop"
path_model_guide=root+os.sep+"test_collect_data"+os.sep
recent_class=0
step_all=[]
step_average=[]
guide_list=[]
path_image=path_model_guide+'guide_video_image'+os.sep
video_name=os.listdir(root+os.sep+'collect_data')
begin=0
if len(video_name)==0:
    begin=0
else:
    for i in range(len(video_name)):
        num_video=int(video_name[i].split('.')[0])
        if begin<num_video:
            begin=num_video

for i in range(6):
    img_guide=cv2.imread(path_image+str(i+2)+".png")
    img_guide = cv2.resize(img_guide, dsize=(650,600))#列，行 ##there ,add some thing  like the color or other thing ,balabala
    guide_list.append(img_guide)
from dataset_scripts.wash_hand import load_data_deploy


def image_process(result, output_data, step_average, recent_class, step_c_t,
                  recent_color):  # shou class and average accuracy
    guide = []
    back_g_all = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 150)
    fontScale = 4
    font_color = (0, 0, 0)
    thickness = 7

    for i in range(6):

        img_guide = guide_list[i]
        back_g = np.zeros((200, img_guide.shape[1], 3), np.uint8)  # np.uint8 挺重要的，有可能数据类型不对
        if i < recent_class:
            if step_average[i] == "Step" + str(i - 2):  # predict is same as the true class
                back_g[:] = (0, 255, 0)
            else:
                back_g[:] = (0, 0, 255)
            write_step = step_average[i]
            back_g = cv2.putText(back_g, write_step, org, font,
                                 fontScale, font_color, thickness, cv2.LINE_AA)
        elif i == recent_class:
            back_g[:] = (0, recent_color, 0)
            write_step = output_data
            back_g = cv2.putText(back_g, write_step, org, font,
                                 fontScale, font_color, thickness, cv2.LINE_AA)
        else:
            back_g[:] = (0, 0, 0)

        back_g_all.append(back_g)
        guide.append(img_guide)
    guide = cv2.hconcat(guide)
    back_g_all = cv2.hconcat(back_g_all)
    add_image = cv2.vconcat([guide, back_g_all])
    result = cv2.resize(result, dsize=(add_image.shape[1], 1500))  # 列，行
    # add rectangle
    r_shape = result.shape
    # (int(r_shape[0]*0.2),int(r_shape[1]*0.2)),(int(r_shape[0]*0.8),int(r_shape[1]*0.8))
    result = cv2.rectangle(result, (500, 300), (3200, 1300), (0, recent_color, 0), 5)  # (col row)
    result = cv2.vconcat([add_image, result])

    # predict_result =

    return result


def classify_frame(path_model_guide,input_length,inputQueue, outputQueue): # only for input a frame ,output a accuracy,that's all
	# keep looping#need loop all of the time !!!!
    X = [] #the input of  lstm :sliding windows 
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    	while True:
    		# check to see if there is a frame in our input queue
            if not inputQueue.empty():
    			# grab the frame from the input queue, resize it, and
    			# construct a blob from it
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                model, model_params = prediction_utils.load_model("./pretrained_models/xdom_summarization", False,
                                                                  loss_name="mixknn_best")
                model_params['use_rotations'] = None
                model_params['skip_frames'] = [1]  # this one is only skip one image ,default is skip 3 image.

                seq_perc = -1
                data_format = model_params['joints_format']

                # 50 guys do 6 action
                def process_output_skelenton_to_list(results):
                    # not sure the type of mediapipe output ,I use this function convert it to array
                    out = ['0'] * 63
                    # Print handedness and draw hand landmarks on the image.
                    if not results.multi_hand_landmarks:
                        out = out
                        # can not find a hand ,initialize to 0
                    else:
                        # only choose the first one hand
                        hand_landmarks = str(results.multi_hand_landmarks[0])
                        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ',
                                                  hand_landmarks)
                        out = hand_landmarks[1:64]
                    return out

                temp_list = inputQueue.get()

                test_frames = []
                for image in temp_list:
                    results = hands.process(image)
                    skeleton_list = process_output_skelenton_to_list(results)
                    if len(set(skeleton_list)) != 1:
                        test_frames.append(skeleton_list)
                # here replace

                if len(test_frames)==3:


                    start=time.time()

                    input_data = np.array(test_frames, dtype=np.float32)

                    if seq_perc == -1: total_data = load_data_deploy.actions_to_samples(
                        load_data_deploy.load_data(input_data, data_format), -1)
                    # if seq_perc == -1: total_data = load_data_deploy.actions_to_samples(load_data_deploy.load_data(data_format), -1)
                    actions_list = total_data[0]
                    total_annotations = actions_list
                    model_params
                    return_sequences = True
                    np.random.seed(0)
                    data_gen = DataGenerator(**model_params)
                    if return_sequences: data_gen.max_seq_len = 0
                    skels_ann = total_annotations
                    test_i = 0
                    def get_pose_features(validation=False):
                        action_sequences = []
                        skels = skels_ann
                        action_sequences.append(data_gen.get_pose_data_v2(skels, validation=validation))
                        if not return_sequences: action_sequences = pad_sequences(action_sequences,
                                                                                  abs(model_params['max_seq_len']),
                                                                                  dtype='float32', padding='pre')
                        return action_sequences
                    action_sequences = get_pose_features(validation=True)

                    action_sequences_augmented = None
                    return_sequences = True
                    t = time.time()

                    model.set_encoder_return_sequences(return_sequences)

                    # Get embeddings from all annotations
                    if return_sequences:
                        # embs = np.array([ model.get_embedding(s[None]).numpy()[0] for s in action_sequences ])

                        embs = [model.get_embedding(s[None]) for s in action_sequences]
                    embs_aug = None
                    sys.stdout.flush
                    if return_sequences: embs = np.array([e[0] for e in embs])
                    embs = embs[0]
                    import joblib
                    knn_from_joblib = joblib.load('trained_knn_1.pkl')
                    # Use the loaded model to make predictions
                    predict_result= knn_from_joblib.predict(embs)

                    predict_result=predict_result.tolist()
                #               result_max=max(predict_result,key=predict_result.count)
                    result_max=predict_result[1]

                    end=time.time()
                    detections = result_max

                   # write the detections to the output queue
                    outputQueue.put(detections)
            
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

if __name__ == '__main__':
    
    # initialize the input queue (frames), output queue (detections),
    # and the list of actual detections returned by the child process
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)
    detections = None
    # construct a child process *indepedent* from our main process of
    # execution
    p = Process(target=classify_frame, args=(path_model_guide,input_length,inputQueue,
    	outputQueue,))
    p.daemon = True
    p.start()
    
    
    #here is the end of process 
    #def set_color_and_score(class_ID):
    #root="E:"+os.sep+"wash_hand_1"
    #use max is better ?? more easy to complete
    #seq=""+os.sep+"" #windows
    
        
    cap = cv2.VideoCapture(0)   #
#    cap = cv2.VideoCapture("/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/HandWashDataset_self/Step6/Step6_24.avi")

    codec = cv2.VideoWriter_fourcc('M','J','P','G')

    #codec = cv2.VideoWriter_fourcc((*'png '))
    #codec = cv2.VideoWriter_fourcc(*'mjpg')

    fps = 30.0 # 指定写入帧率为30
    frameSize = (640, 480) # 指定窗口大小
    # 创建 VideoWriter对象
    out = cv2.VideoWriter(root+os.sep+"collect_data"+os.sep+str(begin+1)+'.avi', codec, fps, frameSize)
    temp_list = deque(maxlen=3)

    start_class = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        out.write(image)


        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        temp_list.append(image)

        if inputQueue.empty():
            inputQueue.put(temp_list)
    	# if the output queue *is not* empty, grab the detections
        if not outputQueue.empty():
                detections = outputQueue.get()
    	# draw the detections on the frame)
        if detections is not None:
            output_data=detections

            step_all.append(output_data) #某一个步骤的所有的accuracy 的
            step_c_t=time.time()-start_class
            if step_c_t>time_gap:
                start_class=time.time()
                recent_class=recent_class+1
                step_average.append(max(step_all, key=step_all.count))
                step_all=[]
                recent_color=225


            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #        if results.multi_hand_landmarks:
     #         for hand_landmarks in results.multi_hand_landmarks:
      #          mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.

            if (step_c_t//3)%2==0:
                recent_color=recent_color-3
            else:
                recent_color=recent_color+3

            result=image_process(cv2.flip(image, 1),output_data,step_average,recent_class,step_c_t,recent_color)

            cv2.imshow('MediaPipe Hands', result)
            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break
            if recent_class>(6-1):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()