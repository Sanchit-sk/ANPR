# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:26:44 2022

@author: Sanchit
"""

import cv2 as cv
import numpy as np
import pytesseract
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import asyncio

LOCATION = "NIT Main Gate"
TRACK_ACTIVITY = "IN"

# PLATE_TEXT_LENGTH = 10
# Helper functions

# This function would reorder the points
# in an order suitable for warp perspective extraction
def reorder (myPoints):
    myPoints = np.reshape(myPoints,(4,2))
    myPointsNew = np.zeros((4,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew.astype(dtype = np.float32)

# This function would take in the canvas image
# and return the list of image portions as mentioned in the 
# boxes array
def boxesImg(img,boxes):
    width,height = 600,120
    result = []
    for box in boxes:
        points = reorder(box)
        resultPoints = np.float32([[0,0],
                                    [width,0],
                                    [0,height],
                                    [width,height]])
        mat = cv.getPerspectiveTransform(points,resultPoints)
        resultImg = cv.warpPerspective(img,mat,(width,height),cv.INTER_LINEAR)
        result.append(resultImg)

    return result

# Function for OCR
def readImage(img, lang_code):
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
  	# set the PSM mode
    options += " --psm {}".format(8)
    # print(pytesseract.__path__)
    # data = pytesseract.image_to_data(img,lang=lang_code,config=options,output_type=pytesseract.Output.DICT)
    # print(data)
    return pytesseract.image_to_string(img,lang=lang_code,config=options).replace("\n","")

def validPlate(plateText):
    # pattern = "^[A-Z]{2}[ -][0-9]{1,2}(?: [A-Z])?(?: [A-Z]*)? [0-9]{4}$"
    # reg = re.compile(pattern)

    return len(plateText) >= 9 and len(plateText) <= 10
    # print(reg.match(plateText)) 

# Takes in the binary image containing 
# possibly rotated text
# The function returns the image having straight text
def deSkewText(binary,img):
    # print(np.mean(binary))
    # In case of black text on white background(most part white)
    # invert the binary image
    if np.mean(binary) > 127:
        binary = np.bitwise_not(binary)

    contours,_ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    textContours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    coords = np.vstack(textContours).squeeze()
    rect = cv.minAreaRect(coords)
    points = cv.boxPoints(rect)
    # print(points)
    cv.drawContours(img,[points.astype(int)],-1,(255,0,0),2)
    # cv.imshow("Rect",img)
    angle = rect[-1]
    # print(rect)
    if angle > 45:
        angle = -(90 - angle)
        
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(binary, M, (w, h),
	flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated
    
# Function that uses image processing techniques to 
# check whether the passed image has a text
# works only with black text on light background number plate
def hasText(img):
    # cv.imshow("Text Check Image",img)
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    
    #Extracting the value channel from HSV for grayscale transformation
    gray = hsv[:,:,2]
    
    #blackhat to enhance dark text on white/bright numberplate/background
    rectKern = cv.getStructuringElement(cv.MORPH_RECT,(13, 5))
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKern,iterations=3)
    #cv.imshow("Morphed",blackhat)
    
    blur = cv.GaussianBlur(blackhat,(5,5),1)
    #cv.imshow("Blurred",blur)
    
    _,thresh = cv.threshold(blackhat,35,255,cv.THRESH_BINARY)
    #cv.imshow("Thresh",thresh)
    
    #Iterations is crictical and kept 2 here
    #To close even when the letters are way far away
    
    myKernel = np.ones((5,5))
    dilated = cv.dilate(thresh,myKernel,iterations=2)
    eroded = cv.erode(dilated,myKernel,iterations=1)
    
    closed = cv.morphologyEx(eroded,cv.MORPH_CLOSE,rectKern,iterations=2)
    # cv.imshow("Morph closed",closed)
    # threshEroded = cv.erode(thresh,myKernel,iterations=1)
    
    # imgHeight,imgWidth = closed.shape
    # totalPixels = imgHeight*imgWidth
    # # print(totalPixels)
    # # print(closeEroded)
    # whitePixels = 0
    # for row in closed:
    #     for px in row:
    #         if px == 255:
    #             whitePixels += 1
                
    # delta = whitePixels/totalPixels
    # return delta >= 0.3,thresh
    # #cv.imshow("Closed Eroded",closeEroded)

    avg = np.mean(closed)
    # print(avg)
    return avg >= 75

# This is the main function that would take in the image/frame
# and return the list of number plates that are present in frame
def extractNumberPlate(gray,img):
    #Reducing the resolution to avoid heavy computation
    # FRAME_WIDTH,FRAME_HEIGHT = 640,480
    # img = cv.resize(img,(FRAME_WIDTH,FRAME_HEIGHT))
    
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray",gray)
    
    #Blurring to reduce the noise
    #That will get rid of some minor edges
    ksize = (5,5)
    # blur = cv.bilateralFilter(gray, 13, 15, 15)
    blur = cv.GaussianBlur(gray,ksize,0)
    edged = cv.Canny(blur, 30, 200) #Edge detection
    cv.imshow("Edged",edged)
    
    #Finding contours and keeping only the top 10 based on the area
    #Thus filtering out small noisy contours
    contours,_ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:20]
    
    #!!! TODO: Improve the defination of rectangle 
    # There are many objects with almost 4 corners 
    # More filters needed
    rectangles = []
    for cnt in contours:
        corners = cv.approxPolyDP(cnt,10,True)
        if len(corners) == 4:
            rectangles.append(corners)
        
    recImages = boxesImg(img,rectangles)

    cv.drawContours(img,rectangles,-1,(255,0,0),2)
    # cv.imshow("Original",img)
    count = 1
    npText = []
    for recImg in recImages:
        textPresent = hasText(recImg) # Pre-detect the possibility of text
        if(textPresent):
            winName = "Box {}".format(count)
            cv.imshow(winName,recImg)
            # sharpened = cv.detailEnhance(recImg,100,0.5)
            
            # Cleaning the image before giving it to OCR input
            # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # sharpened = cv.filter2D(recImg, -1, sharpen_kernel)
            # winName = "Sharpened {}".format(count)
            # cv.imshow(winName,sharpened)
            
            gray = cv.cvtColor(recImg,cv.COLOR_BGR2GRAY)
            
            ret,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            # Deskew the image text to make it straight
            # That would avoid unneccessary errors
            fixed = deSkewText(thresh,recImg)
            winName = "Result {}".format(count)
            cv.imshow(winName,fixed)
            
            # Reading the text from the image
            text = readImage(fixed,"eng")
            npText.append(text)
            count += 1
            # print("Read some plates!")
        
    return npText  

def initFirebase():
    cred = credentials.Certificate('firebase_sdk.json')
    firebase_admin.initialize_app(cred)

def postPlate(plate):
    db = firestore.client()
    # db.collections("plates").document(plate).set({
    #     "time" : "Monday"
    # })

    dateTime = datetime.datetime.now();
    date = dateTime.date()
    time = dateTime.time()
    db.collection("plates").document(str(date)).collection("plates_data").document(str(time)).set({
        "plate": plate,
        "time" : str(time),
        "activity": TRACK_ACTIVITY,
        "location": LOCATION
    })

    print("Plate Posted")

# Image testing driver code
# img = cv.imread("Photos/front/cars1.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Original",img)
# plates = extractNumberPlate(gray,img)
# for plate in plates:
#     print(plate)

# Video testing driver code
initFirebase()
cap = cv.VideoCapture("Videos/gate5.mp4")
BRIGHTNESS = 10
cap.set(10,BRIGHTNESS)

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
frameNumber = 0
while True:
    success,frame = cap.read() 
    if not success: break

    fgmask = fgbg.apply(frame)  
    # cv.imshow("fg mask",fgmask)   

    frameNumber += 1
    if frameNumber == 15: 
        
        # Doing operations on the foreground mask
        sqKern = np.ones((7,7),dtype=np.uint8)
        closed = cv.morphologyEx(fgmask,cv.MORPH_CLOSE,sqKern)
        opened = cv.morphologyEx(closed,cv.MORPH_OPEN,sqKern,iterations=10)
        
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)    
        cv.bitwise_and(gray,opened,gray)
        
        # Extracting the largest moving object in the frame
        # if any object is found the cropping the image
        # to only include that moving object and its nearby surroundings
        contours,_ = cv.findContours(gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
        
        if len(contours):
            x,y,w,h = cv.boundingRect(contours[0])
            x,y = (y,x)
            croppedGray = gray[x:x+h,y:y+w]
            croppedFrame = frame[x:x+h,y:y+w,:]
            
            # print(frame.shape)
            cv.imshow("Largest Moving",croppedFrame)
            plates = extractNumberPlate(croppedGray,croppedFrame)
        
            for plate in plates: 
                if validPlate(plate):
                    print(plate) 
                    # print(f"started at {time.strftime('%X')}")
                    postPlate(plate)
                    # print(f"finished at {time.strftime('%X')}")
                    pass
           
        frameNumber = 0
     
    if success: cv.imshow("Test Video",frame)
    q = cv.waitKey(1)
    if q == ord('q'):
        break

# mask = np.zeros(gray.shape, np.uint8)
# new_image = cv.drawContours(mask, location, -1,255, -1)
# new_image = cv.bitwise_and(img, img, mask=mask)

cv.waitKey(0)
print("Terminating the license plate detection")
cap.release()
cv.destroyAllWindows()