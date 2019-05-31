
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization , Input ,concatenate
from keras.losses import categorical_crossentropy,categorical_hinge,hinge,squared_hinge
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


data_name = {0:'Jaffe',1:'Fer2013',2:'CK+',3:'Big_Fer2013'}



# def load_Big_Fer2013():
#
#
#     # X_Train, Y_Train = Data_PreModeling.mutant('Fer_X.npy', 'Fer_Y.npy')
#     #
#     # X_Valid , Y_Valid = Data_PreModeling.mutant('CK+_X.npy','CK+_Y.npy')
#     #
#     # X_Test , Y_Test = Data_PreModeling.mutant('Jaffe_FaceCrop_X.npy', 'Jaffe_FaceCrop_Y.npy')
#
#     np.save("Big_Fer2013_X_train.npy",X_Train)
#     np.save("Big_Fer2013_X_valid.npy",X_Valid)
#     np.save("Big_Fer2013_X_test.npy",X_Test)
#     np.save("Big_Fer2013_Y_train.npy",Y_Train)
#     np.save("Big_Fer2013_Y_valid.npy",Y_Valid)
#     np.save("Big_Fer2013_Y_test.npy",Y_Test)
#     return X_Train, X_Test, X_Valid, Y_Train, Y_Test, Y_Valid



def load_data_fer2013():


    x = np.load('F:\HAYTD\dataset\Fer_X.npy')
    y = np.load('F:\HAYTD\dataset\Fer_Y.npy')

    x = np.expand_dims(x, -1)
    x = x / 255.0
    y = np.eye(7,dtype='uint8')[y]
    print(y.shape)
    print(y)

    Split = np.load('F:\HAYTD\dataset\Fer_Usage.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')

    X_Train = x[x_index[0]:x_index[-1]+1]
    X_Valid = x[y_index[0]:y_index[-1]+1]
    X_Test  = x[z_index[0]:z_index[-1]+1]
    Y_Train = y[x_index[0]:x_index[-1]+1]
    Y_Valid = y[y_index[0]:y_index[-1]+1]
    Y_Test  = y[z_index[0]:z_index[-1]+1]

    # np.save("Fer2013_X_train.npy",X_Train)
    # np.save("Fer2013_X_valid.npy",X_Valid)
    # np.save("Fer2013_X_test.npy",X_Test)
    # np.save("Fer2013_Y_train.npy",Y_Train)
    # np.save("Fer2013_Y_valid.npy",Y_Valid)
    # np.save("Fer2013_Y_test.npy",Y_Test)

    print(len(Y_Train))
    print(len(Y_Valid))
    print(len(Y_Test))
    return  X_Train,X_Test,X_Valid,Y_Train,Y_Test,Y_Valid

def load_data_Jaffe():


    x = np.load('Jaffe_FaceCrop_X.npy')
    y = np.load('Jaffe_FaceCrop_Y.npy')

    x = np.expand_dims(x, -1)
    x = x / 255.0
    y = np.eye(7,dtype='uint8')[y]

    X_Train = x[0:170]
    X_Valid = x[171:192]
    X_Test  = x[193:213]
    Y_Train = y[0:170]
    Y_Valid = y[171:192]
    Y_Test  = y[193:213]
    print(len(Y_Train))
    print(len(Y_Valid))
    print(len(Y_Test))


    np.save("Jaffe_X_train.npy",X_Train)
    np.save("Jaffe_X_valid.npy",X_Valid)
    np.save("Jaffe_X_test.npy",X_Test)
    np.save("Jaffe_Y_train.npy",Y_Train)
    np.save("Jaffe_Y_valid.npy",Y_Valid)
    np.save("Jaffe_Y_test.npy",Y_Test)

    return  X_Train,X_Test,X_Valid,Y_Train,Y_Test,Y_Valid

def load_data_CKplus():

    x = np.load('CK+_X.npy')
    y = np.load('CK+_Y.npy')

    x = np.expand_dims(x, -1)
    x = x / 255.0
    y = np.eye(7,dtype='uint8')[y]

    Split = np.load('CK+_Usage.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')

    X_Train = x[x_index[0]:x_index[-1]+1]
    X_Valid = x[y_index[0]:y_index[-1]+1]
    X_Test = x[z_index[0]:z_index[-1]+1]
    Y_Train = y[x_index[0]:x_index[-1]+1]
    Y_Valid = y[y_index[0]:y_index[-1]+1]
    Y_Test = y[z_index[0]:z_index[-1]+1]

    np.save("CK+_X_train.npy", X_Train)
    np.save("CK+_X_valid.npy", X_Valid)
    np.save("CK+_X_test.npy", X_Test)
    np.save("CK+_Y_train.npy", Y_Train)
    np.save("CK+_Y_valid.npy", Y_Valid)
    np.save("CK+_Y_test.npy", Y_Test)
    print(len(Y_Train))
    print(len(Y_Valid))
    print(len(Y_Test))


    return  X_Train,X_Test,X_Valid,Y_Train,Y_Test,Y_Valid



def ConvNet_v1(code):



    num_features = 64
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height = 48, 48
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    if code == 0:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_Jaffe()
    if code == 1:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_fer2013()
    if code == 2:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_CKplus()

    model = Sequential()

    model.add(Conv2D(int(num_features/2), kernel_size=(5, 5), activation='relu', input_shape=(width, height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))



    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    filepath = "ConvNetV1_"+data_name[code]+"_best_weights.hdf5"
    early_stop = EarlyStopping(monitor='val_acc', patience=100,mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,early_stop]

    model_json = model.to_json()
    with open("ConvNetV1_" + data_name[code] + "_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow(X_Train,y_Train,
                                            batch_size=batch_size),
                                            steps_per_epoch= len(y_Train) / batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks_list,
                                            validation_data=(X_Valid,y_Valid),
                                            shuffle=True
                        )



    print("Model has been saved to disk ! Training time done !")

def ConvNet_v2(code):

    num_features = 64
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height = 48, 48
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    if code == 0:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_Jaffe()
    if code == 1:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_fer2013()
    if code == 2:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_CKplus()

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(width, height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])



    filepath = "ConvNetV2_"+data_name[code]+"_best_weights.hdf5"
    early_stop = EarlyStopping(monitor='val_loss', patience=20,mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,early_stop]

    model_json = model.to_json()
    with open("ConvNetV2_"+data_name[code]+"_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow(X_Train,y_Train,
                                            batch_size=batch_size),
                                            steps_per_epoch= len(y_Train) / batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks_list,
                                            validation_data=(X_Valid,y_Valid),
                                            shuffle=True
                        )

    print("Model has been saved to disk ! Training time done !")

def ExtractFeatures_Layer(dim):


    model = Sequential()
    model.add(Dense(4096,input_dim=dim,kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    return model


def DSift_ExtractFeatures_Layer(shape):
    model = Sequential()
   
    model.add(Dense(4096, input_shape=shape, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    return model

def CNN_Layer(width, height, depth):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(width, height,depth),
                     data_format='channels_last'))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    return model


def CNN_and_SIFT(code):


    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height , depth = 48, 48 ,1
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    print("Loading Data !")

    if code == 0:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_Jaffe()
    if code == 1:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_fer2013()
    if code == 2:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_CKplus()

    Split = np.load('Fer_Usage_Cropped.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')

    X_SIFT = np.load("Fer2013_SIFTDetector_Histogram.npy")
    X_SIFT = X_SIFT.astype('float64')
    X_SIFT_Train = X_SIFT[x_index[0]:x_index[-1]+1]
    X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1]+1]

    print("Data has been gernerated !")

    SIFT = ExtractFeatures_Layer(X_SIFT_Train.shape[1])
    CNN = CNN_Layer(width,height,depth)

    MergeModel = concatenate([CNN.output, SIFT.output])

    m = Dense(2048, activation='relu')(MergeModel)
    m = Dropout(0.5)(m)
    m = Dense(num_labels, activation='softmax')(m)

    model = Model(inputs=[CNN.input, SIFT.input], outputs=m)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    filepath = "ConvSIFTNET_"+data_name[code]+"_best_weights.hdf5"
    early_stop = EarlyStopping(monitor='val_acc', patience=50,mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,early_stop]

    model_json = model.to_json()
    with open("ConvSIFTNET_"+data_name[code]+"_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow([X_Train,X_SIFT_Train],y_Train,
                batch_size=batch_size),
                steps_per_epoch= len(y_Train)/ batch_size,
                epochs = epochs,
                verbose = 1,
                callbacks = callbacks_list,
                validation_data = ([X_Valid,X_SIFT_Valid],y_Valid),
                shuffle = True
    )


    print("Model has been saved to disk ! Training time done !")


def CNN_and_FAST(code):
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height, depth = 48, 48, 1
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    print("Loading Data !")
    if code == 0:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_Jaffe()
    if code == 1:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_fer2013()
    if code == 2:
        X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_CKplus()


    Split = np.load('Fer_Usage_Cropped.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')
    X_FAST = np.load("Fer2013_FASTDetector_Histogram.npy")
    X_FAST = X_FAST.astype('float64')
    X_FAST_Train = X_FAST[x_index[0]:x_index[-1]+1]
    X_FAST_Valid = X_FAST[y_index[0]:y_index[-1]+1]

    print("Data has been gernerated !")

    FAST = ExtractFeatures_Layer(X_FAST_Train.shape[1])
    CNN = CNN_Layer(width, height, depth)

    MergeModel = concatenate([CNN.output, FAST.output])

    m = Dense(2048, activation='relu')(MergeModel)
    m = Dropout(0.5)(m)
    m = Dense(num_labels, activation='softmax')(m)

    model = Model(inputs=[CNN.input, FAST.input], outputs=m)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    filepath = "ConvFASTNET_" + data_name[code] + "_best_weights.hdf5"
    early_stop = EarlyStopping(monitor='val_acc', patience=50, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stop]

    model_json = model.to_json()
    with open("ConvFASTNET_" + data_name[code] + "_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow([X_Train, X_FAST_Train], y_Train,
                                            batch_size=batch_size),
                        steps_per_epoch=len(y_Train) / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=([X_Valid, X_FAST_Valid], y_Valid),
                        shuffle=True
                        )

    print("Model has been saved to disk ! Training time done !")

# def CNN_SVM(code):
#
#
#     num_labels = 7
#     batch_size = 128
#     epochs = 300
#     width, height = 48, 48
#     data_generator = ImageDataGenerator(
#         featurewise_center=False,
#         featurewise_std_normalization=False,
#         rotation_range=10,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=.1,
#         horizontal_flip=True)
#
#     if code == 0:
#         X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_Jaffe()
#     if code == 1:
#         X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_fer2013()
#     if code == 2:
#         X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data_CKplus()
#
#     model = Sequential()
#
#     model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(width, height, 1),
#                      data_format='channels_last'))
#     model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
#     model.add(Dropout(0.1))
#
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
#     model.add(Dropout(0.1))
#
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
#     model.add(Dropout(0.4))
#
#     model.add(Flatten())
#     model.add(Dense(2048, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_labels,activation='linear',W_regularizer=l2(0.01)))
#
#     model.compile(loss=hinge,
#                   optimizer='adadelta',
#                   metrics=['accuracy'])
#
#     filepath = "ConvNetxSVM_" + data_name[code] + "_best_weights.hdf5"
#     early_stop = EarlyStopping(monitor='val_acc', patience=100, mode='max')
#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint, early_stop]
#
#     model.fit_generator(data_generator.flow(X_Train, y_Train,
#                                             batch_size=batch_size),
#                         steps_per_epoch=len(y_Train) / batch_size,
#                         epochs=epochs,
#                         verbose=1,
#                         callbacks=callbacks_list,
#                         validation_data=(X_Valid, y_Valid),
#                         shuffle=True
#                         )
#
#
#     model_json = model.to_json()
#     with open("ConvNetxSVM_" + data_name[code] + "_model.json", "w") as json_file:
#         json_file.write(model_json)
#
#     print("Model has been saved to disk ! Training time done !")








