import cv2, os, subprocess, time

#Download vmbpy from the manufacturer's repo: https://github.com/alliedvision/VmbPy
from vmbpy import *

#maestro is a file called maestro.py, used to control a Polulu Maestro Servo Driver. Take the file maestro.py at this link and put in the same directory as this code. https://github.com/FRC4564/Maestro/blob/master/maestro.py
import maestro
servo = maestro.Controller('COM4')

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
#To get two DAC boards with different addresses, you will have to put a blob of solder to cover specific pads on the back of the board. Link here, looks for the "I2C Communcation" section. https://learn.sparkfun.com/tutorials/mcp4725-digital-to-analog-converter-hookup-guide#board-overview-
dac = adafruit_mcp4725.MCP4725(i2c, address=0x60)
dac2 = adafruit_mcp4725.MCP4725(i2c, address=0x61)
dac.value=0
dac2.value=0
print("dac!")

#Set exposure time of the camera in seconds. vmbpy takes in us as the exposure value, so this is added for conversion convenience.
exposure_seconds = 1.5
EXPOSURE = exposure_seconds * 1000000

#Set gain here. Gain is measure in dB.
GAIN = 10

#Set where you want to save images to.
path = "YOUR_PATH_HERE/"

#How often you want to wait between each image capture (in seconds)
SLEEP = 5

#The value passed to the LED driver from the DAC is an unsigned 16 bit integer. 65535 = 100%. Scale according to the desired intensity
DAC1_INTENSITY = 65535
DAC2_INTENSITY = 65535

PHOTOINTERVAL = SLEEP
if not os.path.exists(path):
    os.mkdir(path)

def ticcmd(*args):
  return subprocess.check_output(['ticcmd'] + list(args))

i=0

def vimbagrab(path, i, cam, exp, gain, dac):
    exposure_time = cam.ExposureTime
    exposure_time.set(EXPOSURE)
    cam.Gain.set(GAIN)
    tim = exposure_time.get()
    print("exposure is: " + str(tim/1000000)[0:4] + " seconds")
    frame = cam.get_frame(timeout_ms=(exposure_seconds*1000)+4)
    frame.convert_pixel_format(PixelFormat.Mono8)
    print("dac off")
    time.sleep(1)
    print(path+str(i)+'.png')
    cv2.imwrite(path+str(i)+'.png', frame.as_opencv_image())

#Servo positions to line up the excitatory light with a specific dichroic. Calibrate to find your own best values.
servopos1="SOME_INT"
servopos2="SOME_INT"

#Initialize vmbpy camera. Make sure to close any open VimbaXViewer sessions before running the script or it will throw an error.
#All changes to imaging protocol need to go inside the while loop, since the camera is only initialized once. If you want to take a set number of images, change "while True" to some for loop.
#If you want to add position changes in Z while you capture images, add the command 'ticcmd('--exit-safe-start', '--position', str(YOUR_POSITION_NUMBER))' to move the motor to some specific spot.
#Use the ticgui software to find the Z value you want to image at, link: https://www.pololu.com/docs/0J71/3
with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        while True:
            #servopos1 corresponds to a value chosen by calibrating using the Maestro software Polulu supplies here: https://www.pololu.com/docs/0j40/all#3
            #To make it line up with the values given to the maestro.py library, you need to multiply the value by 4.
            servo.setTarget(0,servopos1*4) 
            time.sleep(1)
            x = servo.getPosition(0) #get the current position of servo 1
            print(x)
            dac.value = DAC1_INTENSITY
            time.sleep(2)
            #filenames with "a" will indicate one position, and "b" will indicate the other servo position
            vimbagrab(path, str(i)+"a", cam, EXPOSURE, GAIN, dac)
            time.sleep(1)
            dac.value=0
            time.sleep(2)
            servo.setTarget(0,servopos2*4) 
            time.sleep(1)
            x = servo.getPosition(1) #get the current position of servo 2
            print(x)
            dac2.value = DAC2_INTENSITY
            time.sleep(2)
            vimbagrab(path, str(i)+"b", cam, EXPOSURE, GAIN, dac2)
            time.sleep(1)
            dac2.value=0
            
            
            i+=1
            time.sleep(SLEEP)




servo.close()