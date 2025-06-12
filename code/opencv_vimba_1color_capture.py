import cv2, os, subprocess, time
from vmbpy import *

import board
import busio
import adafruit_mcp4725
i2c = busio.I2C(board.SCL, board.SDA)
dac = adafruit_mcp4725.MCP4725(i2c, address=0x60)
dac.value=0
print("dac!")
     

exposure_seconds = 2
EXPOSURE = exposure_seconds * 1000000

############ THE PART YOU HAVE TO CHANGE ######################
#numcams = 5
#focus_points = [1207,1489,1165,1165,1165,1589]
path = "C:\\Users\\brain\\Documents\\santhosh6_03062025\\"
############ DON'T TOUCH THE REST :) ######################

def ticcmd(*args):
  return subprocess.check_output(['ticcmd'] + list(args))

#if not os.path.exists(path+'cam0'):
    #for x in range(0,numcams+1):
       #os.system("mkdir " + path+ "cam"+str(x))

i=0
#led = digitalio.DigitalInOut(board.C0)
#led.direction = digitalio.Direction.OUTPUT




def vimbagrab(path, i, cam):
    exposure_time = cam.ExposureTime
    exposure_time.set(EXPOSURE)
    cam.Gain.set(5)
    tim = exposure_time.get()
    print("exposure is: " + str(tim/1000000)[0:4] + " seconds")
    dac.value=65000
    print("dac on")
    print(dac.value)
    frame = cam.get_frame(timeout_ms=4000)
    dac.value=0
    frame.convert_pixel_format(PixelFormat.Mono8)
    print(path+str(int(i))+'.png')
    cv2.imwrite(path+str(int(i))+'.png', frame.as_opencv_image())
    print("dac off")

with VmbSystem.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        while True:
            #led.value=False
            #ticcmd('--exit-safe-start', '--position', str(focus_points[int(0)]))
            vimbagrab(path, i, cam)
            
            i+=1
            #time.sleep(5)

            if i>5:
                #led.value=True
                #print(led.value)
                time.sleep(60*60*1)
                #led.value=False
                #print(led.value)
            time.sleep(5)

