# Importing Libraries
import serial
import time
import keyboard

yaw_value = 5
tilt_value = 5


arduino = serial.Serial(port='COM5', baudrate=115200, timeout=15)
builtString = ''

while True:
    time.sleep(0.1)
    if keyboard.is_pressed('right'):
        if yaw_value < 180:
            yaw_value += 5

    if keyboard.is_pressed('left'):
        if yaw_value > 0:
            yaw_value -= 5

    if keyboard.is_pressed('up'):
        if tilt_value < 100:
            tilt_value += 5

    if keyboard.is_pressed('down'):
        if tilt_value > 0:
            tilt_value -= 5

    builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
    arduino.write(builtString.encode())
    arduino.flush()
    print(arduino.readline().decode())



