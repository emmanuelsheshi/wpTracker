import cv2
import serial
import time
import keyboard

# open cv config
faceCascade = cv2.CascadeClassifier("res/haarcascade_frontalface_default.xml")
glassCascade = cv2.CascadeClassifier("res/haarcascade_eye_tree_eyeglasses.xml")
smileCascade = cv2.CascadeClassifier("res/haarcascade_smile.xml")
bodyCascade = cv2.CascadeClassifier("res/haarcascade_upperbody.xml")
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
vid.set(10, 100)  # set the brightness to 150

# tracking parameters
xCp, yCp = 320, 240
xFce, yFce = 0, 0
xEr, yEr = 0, 0
errorThresh = 2

trackingState = 1
pitchTrackingState = 1

# build string to send to microcontroller
builtString = ''
yaw_value = 58
tilt_value = 30

# mode selection
objectDetected = 0
state = 1


def sweepGimbal():
    pass
    # print("Sweeping Camera \n")




while True:
    # time.sleep(0.1)  # this is for smoothening the operation
    ret, frame = vid.read()

    imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGrey, 1.1, 10)
    glasses = glassCascade.detectMultiScale(imgGrey, 1.1, 4)
    body = bodyCascade.detectMultiScale(imgGrey, 1.1, 2)

    # initial values to track
    to_track = faces
    trackParam = "faces"

    # center point for the frame
    cv2.circle(frame, (xCp, yCp), radius=5, color=(0, 0, 255), thickness=-1)

    # ui functions
    if keyboard.is_pressed('1'):
        state = 1

    if keyboard.is_pressed('2'):
        state = 2

    if keyboard.is_pressed('3'):
        state = 3

    if keyboard.is_pressed('r'):
        yaw_value, tilt_value = 58, 30
        print("position reset \n")
        builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
        # arduino.write(builtString.encode())
        # arduino.flush()

    if state == 1:
        to_track = faces
        trackParam = "faces"
        print(f' tracking is set to Face Mode \n')
    elif state == 2:
        to_track = glasses
        trackParam = "Eyes"
        print(f'tracking is set to Eye Mode \n')
    elif state == 3:
        to_track = body
        trackParam = "Body"
        print(f'tracking is set to Body Mode \n')

    # is an object detected or not ?

    if len(to_track) > 0:
        objectDetected = 1
    elif len(to_track) == 0:
        objectDetected = 0

    # if an object is detected

    if objectDetected:
        for (x, y, w, h) in to_track:
            cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0), 2)
            xFce, yFce = x + w // 2, y + w // 2
            cv2.circle(frame, (xFce, yFce), radius=5, color=(0, 255, 0), thickness=-1)

        cv2.line(frame, (xCp, yCp), (xFce, yFce), color=(0, 255, 0))

        xEr = xCp - xFce
        yEr = yCp - yFce

        print(yEr)

        if xEr >= errorThresh:
            if trackingState:
                if xEr > 80:
                    if yaw_value < 180:
                        yaw_value += 5

                if xEr > 50:

                    if yaw_value < 180:
                        yaw_value += 1


        elif xEr < errorThresh:
            if trackingState:
                if yaw_value > 0:
                    yaw_value -= 2

        if yEr <= errorThresh:
            if pitchTrackingState:
                tilt_value += 2
                print("pitch down here \n")

        elif yEr > errorThresh:
            if pitchTrackingState:

                tilt_value -= 2
                print("pitch up here \n")
                if tilt_value <= 0:
                    tilt_value = 0

        # dead band setting for the yaw axis

        if -10 <= xEr <= 10:
            trackingState = 0
        else:
            trackingState = 1

        if -30 <= yEr <= 30:
            pitchTrackingState = 0
        else:
            pitchTrackingState = 1

        builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
        # arduino.write(builtString.encode())
        # arduino.flush()
        # print(arduino.readline().decode())


    # if no object is detected
    else:
        # print(f'no {trackParam}\'s detected \n')
        sweepGimbal()

    cv2.imshow("result", frame)
    cv2.waitKey(10)  # this is to make the video smooth

    if cv2.waitKey(1) & 0xFF == ord('q'):  ### if you press the key q for more than 1 second.....
        break

vid.release()
cv2.destroyWindow("result")