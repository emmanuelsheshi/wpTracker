import cv2
import mediapipe as mp
import time
import serial
import keyboard
import serial.tools.list_ports
import numpy as np

# method to get the arduino port dynamically
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


# call methods to get the ports
foundPorts = get_ports()
connectPort = findArduino(foundPorts)

# polarities left(ccw), right(cw), stall 0,1,2
polYaw = 2


class FaceMesh:
    def __init__(self, static_image_mode=False, max_num_faces=4, refine_landmarks=False,
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


class poseDetector():
    def __init__(self, mode=False, modelComp = 1, smooth = True,
                 enableSeg = False, smoothSeg=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComp = modelComp
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.smooth =    smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smooth, self.enableSeg,
                                            self.smoothSeg, self.detectionCon, self.trackCon)

        # static_image_mode = False,
        # model_complexity = 1,
        # smooth_landmarks = True,
        # enable_segmentation = False,
        # smooth_segmentation = True,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5):




    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) ### its gotten the poses hrer
        if self.results.pose_landmarks:
            if draw:
               self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def  findPositions(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 200), cv2.FILLED)


            return lmList


def main():
    arduino = serial.Serial(connectPort, baudrate=112500, timeout=15)
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




    cap = cv2.VideoCapture(0)
    ctime, ptime = 0, 0
    faceMesh = FaceMesh()
    detector = poseDetector()


    while True:
        # time.sleep(0.005)
        success, img = cap.read()  # read the image

        # ui functions
        if keyboard.is_pressed('1'):
            state = 1
            print("tracking faces")

        elif keyboard.is_pressed('2'):
            state = 2
            print("tracking body")

        elif keyboard.is_pressed('3'):
            state = 3
            print("tracking objects")

        elif keyboard.is_pressed('r'):
            yaw_value, tilt_value, polYaw = 58, 30, 2
            print("position reset \n")
            builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
            arduino.write(builtString.encode())
            arduino.flush()

        # use match instead
        # if face is selected

        if state == 1:
            img, faces, cp_face = faceMesh.detectFaceMesh(img)
            xFce, yFce = cp_face[0], cp_face[1]
            cv2.circle(img, (cp_face[0], cp_face[1]), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img, (xCp, yCp), radius=5, color=(255, 0, 0), thickness=-1)
            xEr = xCp - xFce
            yEr = yCp - yFce

            if len(faces) != 0:
                if xEr >= errorThresh:
                    if trackingState:
                        if yaw_value < 180:
                            yaw_value += 2
                            print("yaw right here")
                            polYaw = 1



                elif xEr < errorThresh:
                    if trackingState:
                        if yaw_value > 0:
                            yaw_value -= 2
                            print("yaw left here")
                            polYaw = 0

                if yEr <= errorThresh:
                    if pitchTrackingState:
                        if tilt_value <= 100:
                            tilt_value += 2
                            # print("pitch down here \n")

                elif yEr > errorThresh:
                    if pitchTrackingState:
                        tilt_value -= 2
                        # print("pitch up here \n")
                        if tilt_value <= 25:
                            tilt_value = 25

                    # dead band setting for the yaw axis

                if -2 <= xEr <= 2:
                    trackingState = 0
                    polYaw = 2

                else:
                    trackingState = 1

                if -20 <= yEr <= 20:
                    pitchTrackingState = 0


                else:
                    pitchTrackingState = 1

                builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + ':' + f'{polYaw}''\n'
                arduino.write(builtString.encode())
                arduino.flush()
                # print(builtString)
                print(arduino.readline().decode())

        # if the body is selected
        if state == 2:
            img = detector.findPose(img, draw=True)
            lmList = detector.findPositions(img, True)


        ctime = time.time()
        frameRate = str(int(1 / (ctime - ptime)))
        ptime = ctime

        # print(cp_face)

        cv2.putText(img, frameRate, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()



