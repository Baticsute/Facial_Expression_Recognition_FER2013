# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HAYTD.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import  QApplication,QFileDialog,QWidget
import cv2
from keras.models import model_from_json
import numpy as np
import pickle
import MakeFeatures
import fnmatch
import source
model_name = {0:'Jaffe' , 1:'Fer2013' , 2 :'CK+',3:'Big_Fer2013'}

json_model = open("models/ConvNetV1_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_CNN_1 = model_from_json(loaded_json_model)
model_CNN_1.load_weights("models/ConvNetV1_Fer2013_best_weights.hdf5")

json_model = open("models/ConvNetV2_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_CNN_2 = model_from_json(loaded_json_model)
model_CNN_2.load_weights("models/ConvNetV2_Fer2013_best_weights.hdf5")

model_CNN_2.summary()

json_model = open("models/ConvSIFTNET_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_SIFTNET = model_from_json(loaded_json_model)
model_SIFTNET.load_weights("models/ConvSIFTNET_Fer2013_best_weights.hdf5")

json_model = open("models/ConvFASTNET_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_FASTNET = model_from_json(loaded_json_model)
model_FASTNET.load_weights("models/ConvFASTNET_Fer2013_best_weights.hdf5")

Kmean_SIFT = pickle.load(open("models/Fer2013_SIFTDetector_Kmean_model.sav", 'rb'))
Kmean_FAST = pickle.load(open("models/Fer2013_FASTDetector_Kmean_model.sav", 'rb'))
emotion_dict = {0: "ANGRY", 1: "DISGUST", 2: "FEAR", 3: "HAPPY", 4: "SAD", 5: "SURPRISE", 6: "NEUTRAL"}

#net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt","models/FaceNet.caffemodel")
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "models/FaceNet.caffemodel"
    configFile = "models/deploy.proto.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def imageDetect(img):

    if (img.shape[0] > 720 or img.shape[1] > 1080):
        if(img.shape[0] > img.shape[1]):
            scale_by_H = 0
        else:
            scale_by_H = 1
        if scale_by_H == 0:
            scale_factor = int(img.shape[0] /  720)
        else:
            scale_factor = int(img.shape[1] / 1080)
        dim = None
        if scale_factor > 1:
            scale_percent = (scale_factor-1) * 10
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
        else:
            scale_percent = 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
        img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300),interpolation=cv2.INTER_AREA), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, ex, ey) = box.astype("int")
            width = x + ex
            height = y + ey

            roi_gray = gray[y:ey, x:ex]
            cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            sift_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_SIFT,"SIFT")
            sift_bow_vector = [sift_bow_vector]
            sift_bow_vector = np.array(sift_bow_vector)

            fast_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_FAST,"FAST")
            fast_bow_vector = [fast_bow_vector]
            fast_bow_vector = np.array(fast_bow_vector)

            cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
            cropped_img = cropped_img / 255.0

            predicted_V1 = model_CNN_1.predict(cropped_img)
            predicted_V2 = model_CNN_2.predict(cropped_img)
            predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])
            predicted_FAST = model_FASTNET.predict([cropped_img, fast_bow_vector])

            predicted_combine = (predicted_SIFT + predicted_FAST + predicted_V1 + predicted_V2) / 4.0

            top = predicted_combine[0].argsort()[-2:][::-1]

            Prob1 = predicted_combine[0][top[0]]
            Prob2 = predicted_combine[0][top[1]]

            fontScale = (width + height) / (width * height) + 0.4

            cv2.rectangle(img, (x - 20, y), (ex + 20, ey), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x - 20, ey), (ex + 20, ey + 20), (0, 255, 0), -1)
            cv2.putText(img, emotion_dict[top[0]], (x - 15, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255),
                        1, cv2.LINE_AA)

            if Prob2 > 0.05:
                cv2.rectangle(img, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), (0, 255, 0), -1)
                cv2.putText(img, emotion_dict[top[1]], (x - 15, ey + 15 + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (0, 0, 255), 1, cv2.LINE_AA)

    # cv2.imwrite(fileout, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imshow('Result', img)


def videoDetect(path):
    cap = cv2.VideoCapture(path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cap.get(3)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cap.get(4)))
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') or cap.isOpened()== 0:
            cap.release()
            cv2.destroyAllWindows()
            return
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        (h, w) = frame.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, ex, ey) = box.astype("int")
                width = x + ex
                height = y + ey

                roi_gray = gray[y:ey, x:ex]
                cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                sift_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_SIFT, "SIFT")
                sift_bow_vector = [sift_bow_vector]
                sift_bow_vector = np.array(sift_bow_vector)

                fast_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_FAST, "FAST")
                fast_bow_vector = [fast_bow_vector]
                fast_bow_vector = np.array(fast_bow_vector)

                cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
                cropped_img = cropped_img / 255.0

                predicted_V1 = model_CNN_1.predict(cropped_img)
                predicted_V2 = model_CNN_2.predict(cropped_img)
                predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])
                predicted_FAST = model_FASTNET.predict([cropped_img, fast_bow_vector])

                predicted_combine = (predicted_SIFT + predicted_FAST + predicted_V1 + predicted_V2) / 4.0

                top = predicted_combine[0].argsort()[-2:][::-1]

                Prob1 = predicted_combine[0][top[0]]
                Prob2 = predicted_combine[0][top[1]]

                fontScale = (width + height) / (width * height) + 0.4

                # cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), (0, 255, 0), -1)
                # cv2.putText(frame, emotion_dict[top[0]], (x - 15, ey + 15),cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), (0, 255, 0), -1)
                # cv2.putText(frame, emotion_dict[top[0]] + " %.f%%" % (Prob1 * 100), (x - 15, ey + 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, emotion_dict[top[0]], (x - 15, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (0, 0, 255), 1, cv2.LINE_AA)
                if Prob2 > 0.05:
                    cv2.rectangle(frame, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), (0, 255, 0), -1)
                    # cv2.putText(frame, emotion_dict[top[1]] + " %.f%%" % (Prob2 * 100), (x - 15, ey + 15 + 15),
                    #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, emotion_dict[top[1]], (x - 15, ey + 15 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('PRESS Q TO EXIT', frame)


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(389, 532)
        Form.setFixedSize(389,532)
        Form.setWindowIcon(QtGui.QIcon('icon.png'))
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(-190, -100, 751, 631))
        self.label.setStyleSheet("image: url(:/newPrefix/haytd.jpg);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.btn_opencamera = QtWidgets.QPushButton(Form)
        self.btn_opencamera.setGeometry(QtCore.QRect(120, 440, 151, 23))
        self.btn_opencamera.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_opencamera.setObjectName("btn_opencamera")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(50, 470, 51, 16))
        self.label_2.setObjectName("label_2")
        self.btn_browse = QtWidgets.QPushButton(Form)
        self.btn_browse.setGeometry(QtCore.QRect(110, 500, 75, 23))
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.btn_browse.setFont(font)
        self.btn_browse.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_browse.setObjectName("btn_browse")
        self.filenameLine = QtWidgets.QLineEdit(Form)
        self.filenameLine.setGeometry(QtCore.QRect(110, 470, 171, 20))
        self.filenameLine.setObjectName("filenameLine")
        self.btn_detect = QtWidgets.QPushButton(Form)
        self.btn_detect.setGeometry(QtCore.QRect(190, 500, 91, 23))
        self.btn_detect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_detect.setObjectName("btn_detect")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        #   Slots :
        self.btn_opencamera.clicked.connect(self.opencamera)
        self.btn_detect.clicked.connect(self.detect)
        self.btn_browse.clicked.connect(self.broswe)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "How Are You ToDay ?"))
        self.btn_opencamera.setText(_translate("Form", "Open Your Camera"))
        self.label_2.setText(_translate("Form", "File path :"))
        self.btn_browse.setText(_translate("Form", "Browse"))
        self.btn_detect.setText(_translate("Form", "Scan and Detect"))

    def opencamera(self):

        cv2.destroyAllWindows()
        stream = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        stream.set(cv2.CAP_PROP_FPS,60.0)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else:
            fps = stream.get(cv2.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        while True:
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stream.release()
                    cv2.destroyAllWindows()
                    return
                ret, frame = stream.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                except:
                    print("Exception Blob Resize")
                    continue
                net.setInput(blob)
                detections = net.forward()
                (h, w) = frame.shape[:2]

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, ex, ey) = box.astype("int")
                        width = x + ex
                        height = y + ey

                        roi_gray = gray[y:ey, x:ex]

                        cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)


                        sift_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_SIFT, "SIFT")
                        sift_bow_vector = [sift_bow_vector]
                        sift_bow_vector = np.array(sift_bow_vector)

                        fast_bow_vector = MakeFeatures.Vetorize_of_An_Image(cropped_img, Kmean_FAST, "FAST")
                        fast_bow_vector = [fast_bow_vector]
                        fast_bow_vector = np.array(fast_bow_vector)

                        cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
                        cropped_img = cropped_img / 255.0

                        predicted_V1 = model_CNN_1.predict(cropped_img)
                        predicted_V2 = model_CNN_2.predict(cropped_img)
                        predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])
                        predicted_FAST = model_FASTNET.predict([cropped_img, fast_bow_vector])

                        predicted_combine = (predicted_SIFT + predicted_FAST + predicted_V1 + predicted_V2) / 4.0

                        top = predicted_combine[0].argsort()[-2:][::-1]

                        Prob1 = predicted_combine[0][top[0]]
                        Prob2 = predicted_combine[0][top[1]]

                        fontScale = (width + height) / (width * height) + 0.4

                        # cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), (0, 255, 0), 1, cv2.LINE_AA)
                        # cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), (0, 255, 0), -1)
                        # cv2.putText(frame, emotion_dict[top[0]], (x - 15, ey + 15),cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), (0, 255, 0), -1)
                        # cv2.putText(frame, emotion_dict[top[0]] + " %.f%%" % (Prob1 * 100), (x - 15, ey + 15),
                        #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, emotion_dict[top[0]], (x - 15, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                        if Prob2 > 0.05:
                            cv2.rectangle(frame, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), (0, 255, 0), -1)
                            # cv2.putText(frame, emotion_dict[top[1]] + " %.f%%" % (Prob2 * 100), (x - 15, ey + 15 + 15),
                            #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.putText(frame, emotion_dict[top[1]], (x - 15, ey + 15 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('PRESS Q TO EXIT', frame)

            except:
                print("Warning : Bound Box Out of Frame !")
                continue

    def broswe(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Browse", "",
                                                  "Image Files (*.png *.jpg *.tiff  *.jpeg);; Video Files (*.avi *.mp4 *.mov)")
        if fileName:
            print(fileName)
        self.filenameLine.setText(fileName)

    def detect(self):

        if fnmatch.fnmatch(self.filenameLine.text(), "*.mp4") or fnmatch.fnmatch(self.filenameLine.text(),
                                                                                 "*.avi") or fnmatch.fnmatch(
                self.filenameLine.text(), "*.mov"):
            videoDetect(self.filenameLine.text())
        else:
            img = cv2.imread(self.filenameLine.text())
            imageDetect(img)


import source

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

