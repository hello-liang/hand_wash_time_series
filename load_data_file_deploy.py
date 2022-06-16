#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:37:22 2021

@author: liang
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
#for hand wash dataset 



# for media pipe 

path_dataset = './datasets/handwash/test_deploy'
joints_inds = { j:i for i,j in enumerate(['WRIST',
'THUMB_CMC',
'THUMB_MCP',
'THUMB_IP',
'THUMB_TIP',
'INDEX_FINGER_MCP',
'INDEX_FINGER_PIP',
'INDEX_FINGER_DIP',
'INDEX_FINGER_TIP',
'MIDDLE_FINGER_MCP',
'MIDDLE_FINGER_PIP',
'MIDDLE_FINGER_DIP',
'MIDDLE_FINGER_TIP',
'RING_FINGER_MCP',
'RING_FINGER_PIP',
'RING_FINGER_DIP',
'RING_FINGER_TIP',
'PINKY_MCP',
'PINKY_PIP',
'PINKY_DIP',
'PINKY_TIP'])}

joints_min_inds = [ joints_inds[j] for j in ['WRIST', 'MIDDLE_FINGER_MCP', 'THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']]
joints_cp_inds = [ joints_inds[j] for j in [ 'WRIST' ] +\
                        ['THUMB_MCP', 'THUMB_IP', 'THUMB_TIP'] +\
                        [ '{}_{}'.format(finger, part) for finger in  ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY' ] \
                         for part in ['MCP', 'PIP', 'DIP', 'TIP']] ]


    

# Load skeletons and labels from original annotation files.
# Transform the joints to the specfied format
def load_data(skels,data_format = 'common_minimal'):


    # remove the zero row ,no matter left or right ?have problem ,because recently only analysis one hand ,so process the skeleton file at first
    skels = skels.reshape((skels.shape[0], 21, 3))

    if data_format == 'common_minimal':
        skels = skels[:,joints_min_inds]
    elif data_format == 'common':
        skels = skels[:,joints_cp_inds]
    total_data = skels

    return total_data
           

# Split skels into different action sequences if seq_len != -1
def actions_to_samples(total_data, seq_len):
    # if seq_len == -1: return total_data

            
    if seq_len == -1: # this one seems means ,only use the data or not
        total_data = [total_data]
    else:
        skels = total_data # this one seems use which ,use the whole of length or whatever
        # samples = np.array_split(skels, (len(skels)//seq_len)+1)
        samples = [ skels[i:i+seq_len] for i in np.arange(0, len(skels), seq_len) ]
        if len(samples[-1]) < seq_len//2: samples = samples[:-1]

        total_data = samples
            
    return total_data






