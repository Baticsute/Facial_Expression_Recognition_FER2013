from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import Data_PreModeling

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
model_name = {0:'Jaffe' , 1:'Fer2013' , 2 :'CK+',3:'Big_Fer2013'}


def predict_X(model,X_filename,y_filename,code,X_SIFT):

    matched_count = 0
    True_Y = []
    Predicted_Y = []
    X = np.load(X_filename)
    Y = np.load(y_filename)
    # X = X[0:100]
    # Y = Y[0:100]
    ##model = load_model("Jaffe")
    predicted_matrix = model.predict([X,X_SIFT])
    predicted_list = predicted_matrix.tolist()
    true_Y_list = Y.tolist()
    for i in range(len(Y)):
        Proba_max = max(predicted_matrix[i])
        current_class = max(true_Y_list[i])
        class_of_Predict_Y = predicted_list[i].index(Proba_max)
        class_of_True_Y = true_Y_list[i].index(current_class)
        True_Y.append(class_of_True_Y)
        Predicted_Y.append(class_of_Predict_Y)
    for i in range(len(true_Y_list)):
        if True_Y[i] == Predicted_Y[i]:
            matched_count = matched_count + 1

    accuracy = (matched_count /len(Y) ) * 100
    print("Accuracy on test set :" + str(accuracy) + "%")
    print()
    np.save(model_name[code]+"_True_y", True_Y)
    np.save(model_name[code]+"_Predict_y",Predicted_Y)

def Test_Combine(code,X,Y):


    json_model = open("ConvNetV1_"+model_name[code]+"_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_CNN_1 = model_from_json(loaded_json_model)
    model_CNN_1.load_weights("1ConvNetV1_"+model_name[code]+"_best_weights.hdf5")
    # model_CNN_1.summary()
    json_model = open("ConvNetV2_"+model_name[code]+"_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_CNN_2 = model_from_json(loaded_json_model)
    model_CNN_2.load_weights("1ConvNetV2_"+model_name[code]+"_best_weights.hdf5")
    # model_CNN_2.summary()

    json_model = open("ConvSIFTNET_"+model_name[code]+"_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_SIFTNET = model_from_json(loaded_json_model)
    model_SIFTNET.load_weights("2ConvSIFTNET_"+model_name[code]+"_best_weights.hdf5")
    # model_SIFTNET.summary()
    json_model = open("ConvFASTNET_"+model_name[code]+"_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_FASTNET = model_from_json(loaded_json_model)
    model_FASTNET.load_weights("1ConvFASTNET_"+model_name[code]+"_best_weights.hdf5")
    # model_FASTNET.summary()
    Split = np.load('Fer_Usage.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')

    X_SIFT = np.load("Fer2013_SIFTDetector_Histogram.npy")
    X_SIFT = X_SIFT.astype('float64')
    print(X_SIFT.shape)

    X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1]+1]
    X_SIFT_Test = X_SIFT[z_index[0]:z_index[-1]+1]

    X_FAST = np.load("Fer2013_FASTDetector_Histogram.npy")
    X_FAST = X_SIFT.astype('float64')
    print(X_FAST.shape)
    X_FAST_Valid = X_FAST[y_index[0]:y_index[-1]+1]
    X_FAST_Test = X_FAST[z_index[0]:z_index[-1]+1]

    predicted_V1 = model_CNN_1.predict(X)
    predicted_V2 = model_CNN_2.predict(X)
    predicted_SIFT = model_SIFTNET.predict([X,X_SIFT_Test])
    predicted_FAST = model_FASTNET.predict([X,X_FAST_Test])
    predicted_combine =    ( predicted_SIFT+predicted_FAST+predicted_V1+predicted_V2)/4.0



    True_Y = []
    Predicted_Y = []
    predicted_list = predicted_combine.tolist()
    true_Y_list = Y.tolist()

    for i in range(len(Y)):
        Proba_max = max(predicted_combine[i])
        current_class = max(true_Y_list[i])
        class_of_Predict_Y = predicted_list[i].index(Proba_max)
        class_of_True_Y = true_Y_list[i].index(current_class)

        True_Y.append(class_of_True_Y)
        Predicted_Y.append(class_of_Predict_Y)

    print("Accuracy on test set :" + str(accuracy_score(True_Y,Predicted_Y)*100) + "%")

    np.save(model_name[code]+"_True_y", True_Y)
    np.save(model_name[code]+"_Predict_y",Predicted_Y)

Y = np.load("Fer2013_Y_test.npy")
X = np.load("Fer2013_X_test.npy")
Test_Combine(1,X,Y)











