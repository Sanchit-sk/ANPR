# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:42:03 2022

@author: Sanchit
"""

import cv2 as cv
import numpy as np

# File to test the background subtraction
cap = cv.VideoCapture("Videos/gate1.mp4")
BRIGHTNESS = 10
cap.set(10,BRIGHTNESS)
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    success,frame = cap.read()
    if frame is None: break

    fgmask = fgbg.apply(frame) 
    cv.imshow("fg Mask",fgmask)
    
    sqKern = np.ones((7,7),dtype=np.uint8)
    
    # Opening to remove small background noise
    # opened = cv.morphologyEx(fgmask,cv.MORPH_OPEN,sqKern)
    # cv.imshow("Opened Mask",opened)
    
    # Dilating to fill the small black pores
    # Near the number plate region
    closed = cv.morphologyEx(fgmask,cv.MORPH_CLOSE,sqKern)
    cv.imshow("Closed mask",closed)
    
    opened = cv.morphologyEx(closed,cv.MORPH_OPEN,sqKern,iterations=10)
    cv.imshow("Opened Mask",opened)
    
    # eroded = cv.erode(closed,(7,7))
    # cv.imshow("Eroded Mask",eroded)
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)    
    cv.bitwise_and(gray,opened,gray)
    contours,_ = cv.findContours(gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
    
    if len(contours) > 0:
        x,y,w,h = cv.boundingRect(contours[0])
        x,y = (y,x)
        croppedGray = gray[x:x+h,y:y+w]
        print(x,y,w,h)
        print(croppedGray.shape)
        if h and w: cv.imshow("cropped Frame",croppedGray)
    
    cv.imshow("frame",gray)
    
    q = cv.waitKey(20)
    if q == ord('q'):
        break

cap.release()
cv.destroyAllWindows()