import pigpio
import picamera
from timeit import default_timer as timer
import socket
import numpy as np
import time
import argparse
import pytz
from datetime import datetime
import os

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--experiment", type = float, default = 1,
                help="desired expereimental procedure")
ap.add_argument("-l", "--light", type = str, default= 'blue',
                help="light source")
args = vars(ap.parse_args())

#experimental set produres
if args['experiment'] == 1:
    prepulse = 4 * 60 #4 minutes before recording (no light baseline)
    lpulse = 8 * 60 # 8 minutes (light pulse)
    interpulse = 0 # 0 (between light pulses)
    cycle = 1 # number of cycles

if args['experiment'] == 2:
    prepulse = 4 * 60 # 4 minutes
    lpulse = 3 * 60 # 3 minutes
    interpulse = 2 * 60 
    cycle = 2

if args['experiment'] == 3:
    prepulse = 0
    lpulse = 2 * 60
    interpulse = 0
    cycle = 1

#initialize handles for TTL and UDP
pi = pigpio.pi() #TTL

try:
    pi.write(18, 0)
    pi.write(12, 0)
    pi.write(4 , 0)
    print('Turned off lights')
except:
    print('Lights are turned off')

#TTL write
def pulse(GPIO, dur):
    pi.write(GPIO, 1) # high
    time.sleep(dur) # in sec
    pi.write(GPIO, 0) # low

path = '/home/pi/Desktop/'

tmstp = datetime.now(pytz.timezone('US/Pacific')).strftime('%Y_%m_%dT%H_%M_%SZ')
fnm = tmstp + '_test_e_' + str(args['experiment']) + '_l_' + str(args['light']) + '.h264'

print('Filename: ' + fnm)
#for loop sending information

print('Starting recording')
pi.write(4 , 1) # IR light
with picamera.PiCamera() as camera: # default is 30 fps
    camera.resolution = (640, 480) # set resolution
    #camera.color_effects = (128,128) # set to black and white
    camera.start_recording(path + fnm) # start recording
    
    camera.start_preview()

    camera.wait_recording(prepulse)

    if args['light'] == 'blue':
        print('Blue light turn on')
        for i in range(0, cycle):
            camera.annotate_text = 'blue'
            pi.write(18, 1)
            camera.wait_recording(lpulse)
            pi.write(18, 0)
            print('Cycle ' + str(i) + ' of' + str(cycle) + ' completed')

            camera.annotate_text = ' '
            camera.wait_recording(interpulse)

    if args['light'] == 'red':
        print('Red light turn on')
        for i in range(0, cycle):
            camera.annotate_text = 'red'    
            pi.write(12, 1)
            camera.wait_recording(lpulse)
            pi.write(12, 0)
            print('Cycle ' + str(i) + ' of' + str(cycle) + ' completed')

            camera.annotate_text = ' '
            camera.wait_recording(interpulse)

    camera.annotate_text = ' '
    camera.stop_recording()

camera.stop_preview()

# pulse(18 ,10) # blue light 
# pulse(12 ,10) # red light 
pi.write(4, 0)

print("Shutting down.")
pi.stop()
# os.system('ffmpeg -framerate 30 -i ' + path + fnm + ' ' + path + fnm[:-4] + 'mp4')
# os.system('rm ' + path + fnm )

#ffmpeg -framerate 30 -i test.h264 output.mp4
