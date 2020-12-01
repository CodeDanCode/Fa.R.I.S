# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:39:18 2020

@author: Daniel
"""

import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import Adafruit_CharLCD as LCD

# Raspberry Pi GPIO pins
GPIO.setmode(GPIO.BCM)
BUTTON_1 = 21
BUTTON_2 = 20
GREEN_LED = 16 # greed LED
RED_LED = 26 #red LED
YELLOW_LED = 19 # yellow LED

RS = 25
EN = 24
DATA4 = 23
DATA5 = 22
DATA6 = 27
DATA7 = 17
BACKLIGHT = 2
COLUMNS = 16
ROWS = 2

#button setup
GPIO.setup(BUTTON_1,GPIO.IN,pull_up_down = GPIO.PUD_UP) #button door indicator
GPIO.setup(BUTTON_2,GPIO.IN,pull_up_down = GPIO.PUD_UP) #push start ignition
#LED setup
GPIO.setup(GREEN_LED,GPIO.OUT)
GPIO.setup(RED_LED,GPIO.OUT)
GPIO.setup(YELLOW_LED,GPIO.OUT)
#LCD setup
lcd = LCD.Adafruit_CharLCD(RS,EN,DATA4,DATA5,DATA6,DATA7,COLUMNS,ROWS,BACKLIGHT)

#camera and facial recognition setup
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 # id's for face recognition
names = ['None','Daniel','Jon'] #names that set to ids
cam = cv2.VideoCapture(0) #camera source. change depending on camera setup
cam.set(3,640) # window width
cam.set(4,480) # window height
minW = 0.1*cam.get(3) # setup min window width
minH = 0.1*cam.get(4) # setup min window height

# global variables
start_ignition = False
text = None
flag1 = False
button_check = False

# opening door
def input_1(channel):
    lcd.clear()
    print("button 1 press")
    global flag1,start_ignition
    # start face detect if true
    if flag1 == False:
        face_detect()
        flag1 = True
    # if face already running
    elif flag1 == True:
        lcd.message("face detection\n already init.")
        time.sleep(3.0)
        lcd.clear()
    else:
        lcd.message("engine running")
        lcd.sleep(3.0)
        lcd.clear


#ignition switch
def input_2(channel):
    lcd.clear()
    print("button 2 press")
    global start_ignition,button_check
    # if the user is identified and 
    # the ignition button hasn't been pressed
    if start_ignition == True and button_check == False:
        print("Started Engine")
        lcd.message("Start Engine")
        time.sleep(3.0)
        lcd.clear()
        lcd.message("Press Button 2\n To Turn Off")
        button_check = True
    else:
        lcd.message("Turn off Engine")
        time.sleep(2)
        lcd.clear()
        lights_off()
        GPIO.cleanup()
        exit()
        
# turns off lights
def lights_off():
    GPIO.output(GREEN_LED,GPIO.LOW)
    GPIO.output(RED_LED,GPIO.LOW)
    GPIO.output(YELLOW_LED,GPIO.LOW)

# Facial detection program
def face_detect():
    lights_off()
    global start_ignition
    #init ready state indicator LED
    GPIO.output(YELLOW_LED,GPIO.HIGH)
    while True:
        # get camera image and preprocess
        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW),int(minH))
            )
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
            #if user is identified by at least 50%
            if(confidence < 50):
                start_ignition = True
                # init known state indicator LED
                GPIO.output(RED_LED,GPIO.LOW)
                GPIO.output(YELLOW_LED,GPIO.LOW)
                GPIO.output(GREEN_LED,GPIO.HIGH)
                id = names[id]
                # convert confidence for window %
                confidence = " {0}".format(round(100 - confidence))
                # show message to screen
                lcd.message("Face Identified\nPress Button 2")

            else:
                #init unknown state indicator LED
                GPIO.output(GREEN_LED,GPIO.LOW)
                GPIO.output(YELLOW_LED,GPIO.LOW)
                GPIO.output(RED_LED,GPIO.HIGH)
                # give id unknown
                id = "unknown"
                # convert confidence for window %
                confidence = " {0}".format(round(100-confidence))
                lcd.message("Face Unknown")
                time.sleep(3.0)
                lcd.clear()
            # draw text box around detected faces    
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        #turn on streaming window
        cv2.imshow('Camera',img)
        
        k = cv2.waitKey(1) & 0xff 
        #if user identified state is true
        if start_ignition == True:
            time.sleep(5)
            break
        
    #close camera and destroy window
    print("Releasing Cam and destroying Windows")
    cam.release()
    cv2.destroyAllWindows()

#button interupt event, debounce button programaticaly 
GPIO.add_event_detect(BUTTON_1,GPIO.RISING,callback= input_1,bouncetime = 200)
GPIO.add_event_detect(BUTTON_2,GPIO.RISING,callback= input_2,bouncetime = 200)

# main loop
while True:
    lights_off()
    lcd.message("   Fa.R.I.S.")
    time.sleep(3.0)
    lcd.clear()
    lcd.message("Press Button 1 \nto Start System")
    print("press button 1 to enter vehicle, press button 2 to start engine")
    #type q to quit program
    text = input("press q to quit: ")
    if text == "q":
        lights_off()
        lcd.clear()
        break
    
GPIO.cleanup()
