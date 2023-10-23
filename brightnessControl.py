import screen_brightness_control as sbc

# get the brightness
brightness = sbc.get_brightness()
# get the brightness for the primary monitor


# set the brightness to 100%
sbc.set_brightness(50)
# set the brightness to 100% for the primary monitor

# show the current brightness for each detected monitor
for monitor in sbc.list_monitors():
    print(monitor, ':', sbc.get_brightness(display=monitor), '%')