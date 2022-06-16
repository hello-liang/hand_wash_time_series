# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
with open(os.path.join("/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/handwash/test_deploy/joint_processed.txt")) as f:
    skels = f.read().splitlines()
skels = np.array([list(map(float, l.split())) for l in skels])
'''

import pickle
import numpy as np
from data_generator import DataGenerator
#import prediction_utils
import load_data_file_deploy
#model_params = prediction_utils.load_model("./pretrained_models/xdom_summarization", False, loss_name="mixknn_best")
f= open('store.pckl','rb')
model_params=pickle.load(f)
f.close()


#in put a matrix ,output a matrix

def skele_augmentation(skels,model_params):

    model_params['use_rotations'] = None
    model_params['skip_frames'] = [1]  # this one is only skip one image ,default is skip 3 image.
    seq_perc = -1
    data_format = 'common_minimal'

    np.random.seed(0)
    if seq_perc == -1: total_data = load_data_file_deploy.actions_to_samples(
        load_data_file_deploy.load_data(skels,data_format), -1)
    # if seq_perc == -1: total_data = load_data_deploy.actions_to_samples(load_data_deploy.load_data(data_format), -1)

    actions_list = total_data[0]
    total_annotations = actions_list
    model_params
    np.random.seed(0)
    data_gen = DataGenerator(**model_params)
    data_gen.max_seq_len = 0
    action_sequences=data_gen.get_pose_data_v2(total_annotations, validation=True)
    return action_sequences


