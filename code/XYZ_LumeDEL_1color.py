import time, os, serial
import cv2, subprocess, yaml, time, math
from datetime import datetime

#Download vmbpy from the manufacturer's repo: https://github.com/alliedvision/VmbPy
from vmbpy import *

#Set where you want to save images to.
writepath = "YOUR_PATH_HERE/"

#Replace these values with the ID associated with each Polulu Tic T825 Stepper Motor Driver you are using.
#Use the ticgui software to find these values, link: https://www.pololu.com/docs/0J71/3
mid_short = "MIDDLE_MOTOR_ID" #middle
mid_top = "TOP_MOTOR_ID" #top
mid_long = "BOTTOM_MOTOR_ID" #bottom

#Set exposure time of the camera in seconds. vmbpy takes in us as the exposure value, so this is added for conversion convenience.
exposure_seconds = 0.15
EXPOSURE = exposure_seconds * 1000000

#Set gain here. Gain is measure in dB.
gain = 10

#Counter variable for counting images taken.
cycle=0

#Set how intense you'd like the light to be using this value. It's a percentage.
INTENSITY = "100"

#On windows this will be a COM port, check Device Manager. Leave the serial settings as is.
PORT = "YOUR_PORT"

ser = serial.Serial(
    port=PORT,
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1   
)

#Sets light intensity to 0.
#If you want to have the light intensity change in other parts of the code, use the command "ser.write(b"IX\r\n")", where X is some number between 0 and 100.
#LumeDEL LED reference is in the manual, link: https://lumedel.com/lumedel-downloads/
ser.write(b"I0\r\n")

#Interface that allows Polulu Tic T825 commandline interface to be used in Python
def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

#Deenergizes the Polulu Tic T825 associated with the passed ID.
def deenergize(mid):
    print("Deenergizing")
    ticcmd('--deenergize', '-d', mid)

#Energizes the Polulu Tic T825 associated with the passed ID.
def energize(mid):
    print("Energizing")
    ticcmd('--energize', '--exit-safe-start', '-d', mid)

#Zeroes the Polulu Tic T825 associated with the passed ID if endstop has been configured.
def zerostage(mid):
    ticcmd('--exit-safe-start', '--home', 'rev', '-d', mid)
    return

#Moves the Polulu Tic T825 associated with the passed ID by relativepos steps. Relativepos can be positive or negative.
def move(mid, relativepos):
    if(relativepos!=0):
        status = yaml.load(ticcmd('-s', '--full', '-d', mid), Loader=yaml.Loader)
        position = status['Current position']
        print("Current position is {}.".format(position))
        new_target = position+relativepos
        
        print("Setting target position to {}.".format(new_target))
        ticcmd('--exit-safe-start', '--position', str(new_target), '-d', mid)
        #time.sleep(abs(relativepos)/100+5)
    return

#Moves the Polulu Tic T825 associated with the passed ID to the specified position.
def moveabs(mid,pos):
    ticcmd('--exit-safe-start', '--position', str(pos), '-d', mid)
    return

#Function built on top of vmbpy for capturing single images.
def vimbagrab(path, i, cam):
    exposure_time = cam.ExposureTime
    exposure_time.set(EXPOSURE)
    cam.Gain.set(gain)
    timet = exposure_time.get()
    print("exposure is: " + str(timet/1000000)[0:4] + " seconds")
    ser.write(b"I"+INTENSITY+"\r\n")
    time.sleep(1)
    frame = cam.get_frame()
    frame.convert_pixel_format(PixelFormat.Mono8)
    ser.write(b"I0\r\n")
    print(path+'.png')
    cv2.imwrite(path+'.png', frame.as_opencv_image())

with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        while True:
            #Implement your scanning protocol here.
            #Example which turns the motors on, zeroes, them, moves to 1000,1000,1000 and takes a picture.

            '''
            energize(mid_top)
            energize(mid_short)
            energize(mid_long)
            
            zerostage(mid_top)
            time.sleep(30)
            zerostage(mid_short)
            time.sleep(30)
            zerostage(mid_long)
            time.sleep(30)

            moveabs(mid_long, 1000)
            time.sleep(30)
            moveabs(mid_short, 1000)
            time.sleep(30)
            moveabs(mid_top, 1000)
            time.sleep(30)

            path = writepath + str(cycle) + "/" testimage
            try:
                vimbagrab(path, cycle, cam)
            except Exception as e:
                print(e)
            time.sleep(2)
            
            cycle+=1
            zerostage(mid_top)
            deenergize(mid_short)
            deenergize(mid_top)
            deenergize(mid_long)
            time.sleep(60)
            '''



