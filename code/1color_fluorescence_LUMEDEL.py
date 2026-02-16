import cv2, os, subprocess, time

#Download vmbpy from the manufacturer's repo: https://github.com/alliedvision/VmbPy
from vmbpy import *

import serial, time

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

if ser.is_open:
    print("Serial port opened successfully")

#This line makes the LED beep as a test that it is working. Comment out if you don't like the beep.
ser.write(b"b\r\n")

#Set how intense you'd like the light to be using this value. It's a percentage.
#If you want to have the light intensity change in other parts of the code, use the command "ser.write(b"IX\r\n")", where X is some number between 0 and 100.
#LumeDEL LED reference is in the manual, link: https://lumedel.com/lumedel-downloads/
INTENSITY = "100"

#Set exposure time of the camera in seconds. vmbpy takes in us as the exposure value, so this is added for conversion convenience.
exposure_seconds = 1.5
EXPOSURE = exposure_seconds * 1000000

#Set gain here. Gain is measure in dB.
GAIN = 10

#Set where you want to save images to.
path = "YOUR_PATH_HERE/"

#How often you want to wait between each image capture (in seconds)
SLEEP = 5

#Interface that allows Polulu Tic T825 commandline interface to be used in Python
def ticcmd(*args):
  return subprocess.check_output(['ticcmd'] + list(args))

#Counter variable for counting images taken.
i=0

#Function built on top of vmbpy for capturing single images.
def vimbagrab(path, i, cam):
    exposure_time = cam.ExposureTime
    exposure_time.set(EXPOSURE)
    cam.Gain.set(GAIN)
    tim = exposure_time.get()
    print("exposure is: " + str(tim/1000000)[0:4] + " seconds")
    ser.write(b"I"+INTENSITY+"\r\n")
    print("dac on")
    print(dac.value)
    frame = cam.get_frame(timeout_ms=(exposure_seconds*1000)+4)
    ser.write(b"I0\r\n")
    frame.convert_pixel_format(PixelFormat.Mono8)
    print(path+str(int(i))+'.png')
    cv2.imwrite(path+str(int(i))+'.png', frame.as_opencv_image())
    print("dac off")

#Initialize vmbpy camera. Make sure to close any open VimbaXViewer sessions before running the script or it will throw an error.
#All changes to imaging protocol need to go inside the while loop, since the camera is only initialized once. If you want to take a set number of images, change "while True" to some for loop.
#If you want to add position changes in Z while you capture images, add the command 'ticcmd('--exit-safe-start', '--position', str(YOUR_POSITION_NUMBER))' to move the motor to some specific spot.
#Use the ticgui software to find the Z value you want to image at, link: https://www.pololu.com/docs/0J71/3
with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        while True:

            vimbagrab(path, i, cam)
            
            i+=1

            time.sleep(SLEEP)


            

