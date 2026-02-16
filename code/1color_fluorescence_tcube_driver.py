import cv2, os, subprocess, time

#Download vmbpy from the manufacturer's repo: https://github.com/alliedvision/VmbPy
from vmbpy import *

#'board' will not import. Follow this guide to set up your system (Windows, Mac, or Linux) to be used with an FT232H board: https://learn.adafruit.com/circuitpython-on-any-computer-with-ft232h/setup
#You also need to set a environment variable called BLINKA_FT232H and set it to 1
#Windows Powershell: $env:BLINKA_FT232H = 1
#Linux export BLINKA_FT232H = 1
#Make sure the "I2C mode" switch on the FT232H board is set to ON
import board
import busio
import adafruit_mcp4725
i2c = busio.I2C(board.SCL, board.SDA)

#The address of your MCP4725 board might be different than mine. You can scan your system for i2c devices using this script from Adafruit: https://learn.adafruit.com/scanning-i2c-addresses/circuitpython
dac = adafruit_mcp4725.MCP4725(i2c, address=0x60)
dac.value=0
print("dac!")
     
#Set exposure time of the camera in seconds. vmbpy takes in us as the exposure value, so this is added for conversion convenience.
exposure_seconds = 1.5
EXPOSURE = exposure_seconds * 1000000

#Set gain here. Gain is measure in dB.
GAIN = 10

#Set where you want to save images to.
path = "YOUR_PATH_HERE/"

#LED intensity is stored as a 16-bit number. 32767 corresponds to 100% intensity.
LED_INTENSITY = 32767

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
    dac.value=32767
    print("dac on")
    print(dac.value)
    frame = cam.get_frame(timeout_ms=(exposure_seconds*1000)+4)
    dac.value=0
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

