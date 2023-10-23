import cv2



cap = cv2.VideoCapture(0)

while True:
    flag, img  = cap.read()

    cv2.imshow('img', img)
    cv2.waitKey(1)