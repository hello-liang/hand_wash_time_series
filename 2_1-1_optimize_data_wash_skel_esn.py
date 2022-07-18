#still need to do ,1, shuffle

num_frame_analysis=20
part = 0.3
skip=2

import numpy as np
from data_generator import DataGenerator
#import prediction_utils
import load_data_file_deploy
#model_params = prediction_utils.load_model("./pretrained_models/xdom_summarization", False, loss_name="mixknn_best")
import pickle
'''
f= open('store.pckl','wb')
pickle.dump(model_params,f)
f.close()
'''
f= open('store.pckl','rb')
model_params=pickle.load(f)
f.close()

from skel_aug import skele_augmentation
import time

from pandas import read_csv

import random




def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    data = dataframe.values
    # remove o at first
    return dataframe.values


def data_AUG_identify_one_or_two(new_sample,model_params):
    # ndarray
    if (new_sample.shape[1] == 63):  # use for one hand
        data_AUG = np.float64(skele_augmentation(new_sample, model_params))
    else: # 2 hand
        data_AUG_left = np.float64(skele_augmentation(new_sample[:,0:63], model_params))
        data_AUG_right = np.float64(skele_augmentation(new_sample[:,63:126], model_params))
        data_AUG = np.concatenate((data_AUG_left, data_AUG_right), axis=1)
    return data_AUG

def get_train_test_data(path, train_list, test_list):
    # still need to do ,1, shuffle
    trainX = []
    trainy = []
    testX = []
    testy = []

    ###input is train_list
    import os
    data = os.listdir(path)
    for subj in data:
        for step in os.listdir(path + os.sep + subj):
            if(len(step)==5):
                y_class = int(step[-1]) - 1
            elif(len(step)==1):
                y_class = int(step[0]) - 1

            else:
                y_class = int(step[5]) - 1


            if int(subj) in train_list:
                data = load_file(path + os.sep + subj + os.sep + step + os.sep + "joint.txt")  # ndarray [405,63]
                for i in range(data.shape[0]):
                    if (i >= num_frame_analysis) & (random.random() < part):
                        new_sample = data[(i - num_frame_analysis):i:skip, :]
                        data_AUG = data_AUG_identify_one_or_two(new_sample, model_params)

                        trainX.append(data_AUG)
                        trainy.append(y_class)
            if int(subj) in test_list:
                data = load_file(path + os.sep + subj + os.sep + step + os.sep + "joint.txt")
                for i in range(data.shape[0]):
                    if (i >= num_frame_analysis) & (random.random() < part):
                        new_sample = data[(i - num_frame_analysis):i:skip, :]
                        data_AUG = data_AUG_identify_one_or_two(new_sample, model_params)

                        testX.append(data_AUG)
                        testy.append(y_class)
    trainX = np.array(trainX)
    testX = np.array(testX)
    testy = np.array(testy)
    trainy = np.array(trainy)
    return trainX,testX,testy,trainy

def process_esn_train_test(trainX,testX,testy,trainy):
    Xtr = trainX  # shape is [N,T,V]
    Ytr = trainy.reshape((trainy.shape[0], 1))  # shape is [N,1]
    Xte = testX
    Yte = testy.reshape((testy.shape[0], 1))
    print('Loaded ' + ' - Tr: ' + str(Xtr.shape) + ', Te: ' + str(Xte.shape))
    # One-hot encoding for labels
    onehot_encoder = OneHotEncoder(sparse=False)
    Ytr = onehot_encoder.fit_transform(Ytr)
    Yte = onehot_encoder.transform(Yte)
    return Xtr,Ytr,Xte,Yte


def run_optimize():
    start = time.time()

    # ============ Initialize, train and evaluate the RC model ============
    classifier = RC_model(
        reservoir=None,
        n_internal_units=config['n_internal_units'],
        spectral_radius=config['spectral_radius'],
        leak=config['leak'],
        connectivity=config['connectivity'],
        input_scaling=config['input_scaling'],
        noise_level=config['noise_level'],
        circle=config['circ'],
        n_drop=config['n_drop'],
        bidir=config['bidir'],
        dimred_method=config['dimred_method'],
        n_dim=config['n_dim'],
        mts_rep=config['mts_rep'],
        w_ridge_embedding=config['w_ridge_embedding'],
        readout_type=config['readout_type'],
        w_ridge=config['w_ridge'],
        mlp_layout=config['mlp_layout'],
        num_epochs=config['num_epochs'],
        w_l2=config['w_l2'],
        nonlinearity=config['nonlinearity'],
        svm_gamma=config['svm_gamma'],
        svm_C=config['svm_C']
    )
    # ============ Load dataset ============

    path = "skeleton_me_jianxhee_2_hand"
    all_subject = list(range(1, 11))
    random.shuffle(all_subject)
    accuracy_all = []
    f1_all = []
    for i in range(5):
        test_list = all_subject[(i * 2):((i * 2) + 2)]
        train_list = list(set(all_subject) - set(test_list))

        trainX, testX, testy, trainy = get_train_test_data(path, train_list, test_list)
        Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX, testX, testy, trainy)

        tr_time = classifier.train(Xtr, Ytr)
        accuracy, f1, pred_class = classifier.test(Xte, Yte)
        #   pred_class = classifier.predict(Xte)
        accuracy_all.append(accuracy)
        f1_all.append(f1)
    avg_acc = sum(accuracy_all) / 5
    avg_f1 = sum(f1_all) / 5
    # esn_hand_wash_kaggle()
    print(avg_acc)
    print(avg_f1)
    cost_time=time.time()- start
    return avg_acc,avg_f1,cost_time


# test ESNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNn


# General imports
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder
# Custom imports
from modules import RC_model


# grid search :

search_n_internal_units=[300,350,400,450,500,550]

search_Dim_reduce=[ 'pca', 'tenpca'] #have None
search_MTS_representation=['last', 'mean', 'output', 'reservoir']
search_Ty_readout=['lin', 'mlp', 'svm']

# Append 'hello' at the end of file

for Dim_reduce in search_Dim_reduce:
    for MTS_representation in search_MTS_representation:
        for Ty_readout in search_Ty_readout:
            file_object = open('optimize_result.txt', 'a')

            # ============ RC model configuration and hyperparameter values ============
            config = {}
            config['dataset_name'] = 'hand_wash'

            config['seed'] = 1
            np.random.seed(config['seed'])

            # Hyperarameters of the reservoir
            config['n_internal_units'] = 450        # size of the reservoir
            config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
            config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
            config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
            config['input_scaling'] = 0.1           # scaling of the input weights
            config['noise_level'] = 0.01            # noise in the reservoir state update
            config['n_drop'] = 5                    # transient states to be dropped
            config['bidir'] = True                  # if True, use bidirectional reservoir
            config['circ'] = False                  # use reservoir with circle topology

            # Dimensionality reduction hyperparameters
            config['dimred_method'] =Dim_reduce     # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
            config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

            # Type of MTS representation
            config['mts_rep'] = MTS_representation         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
            config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

            # Type of readout
            config['readout_type'] = Ty_readout         # readout used for classification: {'lin', 'mlp', 'svm'}

            # Linear readout hyperparameters
            config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

            # SVM readout hyperparameters
            config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
            config['svm_C'] = 5.0                   # regularization for SVM hyperplane

            # MLP readout hyperparameters
            config['mlp_layout'] = (10,10)          # neurons in each MLP layer
            config['num_epochs'] = 2000             # number of epochs
            config['w_l2'] = 0.001                  # weight of the L2 regularization
            config['nonlinearity'] = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

            print(config['mts_rep'])
            avg_acc,avg_f1,cost_time=run_optimize()
            file_object.write(str(Dim_reduce)+"\t"+str(MTS_representation)+"\t"+str(Ty_readout)+"\t"+str(avg_acc) +"\t"+ str(avg_f1)+"\t"+str(cost_time)+"\n")
            # Close the file

            file_object.close()








