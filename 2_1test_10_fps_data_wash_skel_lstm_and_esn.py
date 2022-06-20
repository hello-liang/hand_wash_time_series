#still need to do ,1, shuffle


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

#load data from handwashdataset_self_one_hand
import random
# lstm model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import random
part = 0.3
num_frame_analysis=30
skip=3


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    data = dataframe.values
    # remove o at first
    return dataframe.values



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
                    if (i >= 30) & (random.random() < part):
                        new_sample = data[(i - 30):i:skip, :]

                        #ndarray
                        data_AUG = np.float64(skele_augmentation(new_sample, model_params))
                        trainX.append(data_AUG)
                        trainy.append(y_class)
            if int(subj) in test_list:
                data = load_file(path + os.sep + subj + os.sep + step + os.sep + "joint.txt")
                for i in range(data.shape[0]):
                    if (i >= 30) & (random.random() < part):
                        new_sample = data[(i - 30):i:skip, :]
                        data_AUG = np.float64(skele_augmentation(new_sample, model_params))

                        testX.append(data_AUG)
                        testy.append(y_class)
    trainX = np.array(trainX)
    testX = np.array(testX)
    testy = np.array(testy)
    trainy = np.array(trainy)
    return trainX,testX,testy,trainy

def process_lstm_train_test(trainX,testX,testy,trainy):
    trainX = np.array(trainX)
    testX = np.asarray(testX)
    testy = np.array(testy)
    trainy = np.array(trainy)
    trainy = trainy
    testy = testy
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
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




def train_lstm(trainX,trainy):
    # fit and evaluate a model
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model,batch_size
# evaluate model
def lstm_HandWashDataset_selfbatch_1_one_hand():
    print("begin to test batch 1")
    path="HandWashDataset_self_one_hand"
    all_subject = list(range(1, 51))
    random.shuffle(all_subject)
    train_list=all_subject[0:35]
    test_list=all_subject[35:50]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    trainX,testX,testy,trainy=process_lstm_train_test(trainX,testX,testy,trainy)
    model,batch_size=train_lstm(trainX,trainy)         ##########only position for train
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    accuracy = accuracy * 100.0
    print('>#%d: %.3f' % (0 + 1, accuracy))

def lstm_collect_data_batch_3_one_hand():

    print("begin to test batch_3_data")
    path = "collect_data_batch_3_one_hand"
    all_subject = list(range(1, 12))
    random.shuffle(all_subject)
    train_list = all_subject[0:6]
    test_list = all_subject[6:12]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    trainX,testX,testy,trainy=process_lstm_train_test(trainX,testX,testy,trainy)
    model,batch_size=train_lstm(trainX,trainy)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    accuracy = accuracy * 100.0
    print('>#%d: %.3f' % (0 + 1, accuracy))

# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
#model.save("my_h5_model.h5")
# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_h5_model.h5")


'''
print("Generate a prediction")
prediction = model.predict(testX[:10])
print(prediction)
for i in range(prediction.shape[0]):
    print(np.argmax(prediction[i,:]))
    print("true")
    print(np.argmax(testy[i,:]))


print(testy[:10])
pred_name = np.argmax(prediction)
print(pred_name)
print("prediction shape:", prediction.shape)
'''




# test ESNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNn


# General imports
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder
# Custom imports
from modules import RC_model
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
config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'mlp', 'svm'}

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

print(config)

# ============ Initialize, train and evaluate the RC model ============
classifier =  RC_model(
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
#data = scipy.io.loadmat('../dataset/'+config['dataset_name']+'.mat')

# data_batch_1

def wsn_HandWashDataset_self_batch_1_one_hand():
    print("begin to test batch 1")

    path="HandWashDataset_self_one_hand"
    all_subject = list(range(1, 51))
    random.shuffle(all_subject)
    train_list=all_subject[0:35]
    test_list=all_subject[35:50]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)

    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

    tr_time = classifier.train(Xtr, Ytr) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!important only position to train
    accuracy, f1, pred_class= classifier.test(Xte, Yte)
    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

def only_test_wsn_HandWashDataset_self_batch_1_one_hand():
    print("begin to test batch 2")

    path="skeleton_ppum_batch_2"
    all_subject = list(range(1, 51))
    random.shuffle(all_subject)
    train_list=all_subject[0:35]
    test_list=all_subject[35:50]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)

    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

#    tr_time = classifier.train(Xtr, Ytr) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!important only position to train
    accuracy, f1, pred_class= classifier.test(Xte, Yte)
    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

def esn_collect_data_batch_3_one_hand():
    print("esn_begin to test batch_3_data")
    path = "skeleton_ppum_batch_3"
    all_subject = list(range(1, 12))
    random.shuffle(all_subject)
    train_list = all_subject[0:6]
    test_list = all_subject[6:12]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

    tr_time = classifier.train(Xtr, Ytr)
    accuracy, f1, pred_class = classifier.test(Xte, Yte)
    pred_class = classifier.predict(Xte)

    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

def esn_hand_wash_kaggle():
    print("esn_hand_wash_kaggle")
    path = "skeleton_hand_wash_kaggle_one"
    all_subject = list(range(1, 26))
    random.shuffle(all_subject)
    train_list = all_subject[0:20]
    test_list = all_subject[20:25]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

    tr_time = classifier.train(Xtr, Ytr)
    accuracy, f1, pred_class = classifier.test(Xte, Yte)
    pred_class = classifier.predict(Xte)

    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))
def only_test_esn_hand_wash_kaggle():
    print("esn_hand_wash_kaggle")
    path = "skeleton_hand_wash_kaggle_one"
    all_subject = list(range(1, 26))
    random.shuffle(all_subject)
    train_list = all_subject[0:20]
    test_list = all_subject[20:25]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

#    tr_time = classifier.train(Xtr, Ytr)
    accuracy, f1, pred_class = classifier.test(Xte, Yte)
    pred_class = classifier.predict(Xte)

    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

def esn_hand_J_L_one():
    path = "skeleton_me_jianxhee"
    all_subject = list(range(1, 10))
    random.shuffle(all_subject)
    train_list = all_subject[0:8]
    test_list = all_subject[8:10]
    trainX,testX,testy,trainy=get_train_test_data(path, train_list, test_list)
    Xtr, Ytr, Xte, Yte = process_esn_train_test(trainX,testX,testy,trainy)

    tr_time = classifier.train(Xtr, Ytr)
    accuracy, f1, pred_class = classifier.test(Xte, Yte)
    pred_class = classifier.predict(Xte)

    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))


#esn_hand_wash_kaggle()
esn_hand_J_L_one()
only_test_esn_hand_wash_kaggle()
only_test_wsn_HandWashDataset_self_batch_1_one_hand()
f= open('classifier.pckl', 'wb')
pickle.dump(classifier,f)
f.close()








