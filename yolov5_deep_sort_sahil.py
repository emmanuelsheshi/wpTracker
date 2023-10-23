import cv2
import mediapipe as mp
import time
import serial
import keyboard
import serial.tools.list_ports
from keras.models import load_model

import pyttsx3
engine = pyttsx3.init() # object creation

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate', 125)     # setting up new voice rate

"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level
engine.setProperty('volume', 1.0)    # setting up volume level  between 0 and 1



# Load the model
model = load_model('keras_model.h5')
# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()



def get_ports():
    ports = serial.tools.list_ports.comports()

    return ports


def findArduino(portsFound):
    commPort = 'None'
    numConnection = len(portsFound)

    for i in range(0, numConnection):
        port = foundPorts[i]
        strPort = str(port)

        if 'CH340' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])

    return commPort
foundPorts = get_ports()
connectPort = findArduino(foundPorts)

#weapond detection
import cv2
import numpy as np
from keras.models import load_model

from threading import Thread


from subprocess import call
phrase = "hostile detected!!!"


# Load the model
model = load_model('keras_model.h5')
# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()
text = "hostile person detected, tracking faces"

def say(message):
    engine.say(message)
    try:
        engine.runAndWait()

    except :
        print("no whalla")
    engine.stop()


class FaceMesh:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

    def detectFaceMesh(self, img, draw=True):
        # img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRgb)
        faces = []
        x, y = 0, 0
        xN, yN = 0, 0

        if self.results.multi_face_landmarks:

            xm = self.results.multi_face_landmarks[0].landmark[0].x
            ym = self.results.multi_face_landmarks[0].landmark[0].y
            iwL, ihL, icL = img.shape
            xN, yN = int(xm * 640), int(ym * 480)

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec,
                                               self.drawSpec)

                face = []

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])

                faces.append(face)

        return img, faces, (xN, yN)


def main():
    arduino = serial.Serial(connectPort, baudrate=115200, timeout=15)

    xCp, yCp = 320, 240
    xFce, yFce = 0, 0
    xEr, yEr = 0, 0
    errorThresh = 2

    trackingState = 1
    pitchTrackingState = 1

    # build string to send to microcontroller
    builtString = ''
    yaw_value = 100
    tilt_value = 58

    # mode selection
    objectDetected = 0
    state = 1

    # cap = cv2.VideoCapture('videos/1.mp4')
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    ctime, ptime = 0, 0
    faceMesh = FaceMesh()

    while True:
        # time.sleep(0.005)
        success, img = cap.read()
        # img2 =  img.copy()
        suc2, img2 = cap2.read()

        #pass the image to the classifier first
        imageClass = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow('Webcam Image', imageClass)

        # Make the image a numpy array and reshape it to the models input shape.
        imageClass = np.asarray(imageClass, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        imageClass = (imageClass / 127.5) - 1
        # Have the model predict what the current image is. Model.predict
        # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
        # it is the first label and 80% sure its the second label.
        probabilities = model.predict(imageClass)
        # Print what the highest value probabilitie label
        # print(labels[np.argmax(probabilities)])

        myLabels = labels[np.argmax(probabilities)]

        print(len(myLabels))


        img, faces, cp_face = faceMesh.detectFaceMesh(img)
        xFce, yFce = cp_face[0], cp_face[1]

        cv2.circle(img, (cp_face[0], cp_face[1]), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, (xCp, yCp), radius=5, color=(255, 0, 0), thickness=-1)

        xEr = xCp - xFce
        yEr = yCp - yFce


        # print(yEr)

        if keyboard.is_pressed('r'):
            yaw_value, tilt_value = 58, 30
            print("position reset \n")
            builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
            arduino.write(builtString.encode())
            arduino.flush()

        if "knives" in myLabels:
            print("knives deteted")
            cv2.putText(img, "Hostile Detected", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Tracking face", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
            # Thread(target=say, args=(text,)).start()



            if len(faces) != 0:

                pass
                if xEr >= errorThresh:
                    if trackingState:
                       if yaw_value < 180:
                        yaw_value += 1


                elif xEr < errorThresh:
                    if trackingState:
                        if yaw_value > 0:
                            yaw_value -= 1

                if yEr <= errorThresh:
                    if pitchTrackingState:
                        if tilt_value <= 100:

                            tilt_value += 2
                            print("pitch down here \n")

                elif yEr > errorThresh:
                    if pitchTrackingState:

                        tilt_value -= 2
                        print("pitch up here \n")
                        if tilt_value <= 25:
                            tilt_value = 25

                    # dead band setting for the yaw axis

                if -50 <= xEr <= 50:
                    trackingState = 0
                else:
                    trackingState = 1

                if -40 <= yEr <= 40:
                    pitchTrackingState = 0
                else:
                    pitchTrackingState = 1

                builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
                arduino.write(builtString.encode())
                arduino.flush()
                print(arduino.readline().decode())

        else:
            pass


        ctime = time.time()
        frameRate = str(int(1 / (ctime - ptime)))
        ptime = ctime

        # print(cp_face)

        cv2.putText(img, frameRate, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()



