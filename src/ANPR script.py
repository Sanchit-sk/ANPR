# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:26:44 2022

@author: Sanchit
"""

import cv2 as cv
import numpy as np
import pytesseract
from firebase_db import initFirebase, postPlate

CAPTURE_BRIGHTNESS = 10
RESOURCE_PATH = "../Videos/gate5.mp4" # Mention the path of the test video/ image here 

def reorder (myPoints):
    """
    This function would reorder the points in an order suitable for warp perspective extraction
    
    @param myPoints: The points list to reorder
    @returns: A list of re-ordered points in float data type
    """

    myPoints = np.reshape(myPoints,(4,2))
    myPointsNew = np.zeros((4,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew.astype(dtype = np.float32)

#######################################################################################

def boxesImg(img,boxes):
    """
    Function to return the list of images in the given image based 
    on the list of box co-ordinates passed

    @param img: The complete image
    @param boxes: List of boxes' corners points to be taken out from img
    @return: List of image portions from img corresponding to every box co-ordinates  
    """

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

#######################################################################################

def readImage(img, lang_code):
    """
    Function to OCR the image passed

    @param img: The image to read the string from
    @param lang_code: Language code of the text in the image
    @returns: The extracted text from the image in string format
    """

    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)

  	# set the PSM mode
    options += " --psm {}".format(8)
    return pytesseract.image_to_string(img,lang=lang_code,config=options).replace("\n","")

#######################################################################################

def validPlate(plateText):
    """
    Function to test whether the number plate text is valid

    @param plateText: The text to validate
    @returns: True if the plate text is correct, false otherwise
    """

    return len(plateText) >= 9 and len(plateText) <= 10

#######################################################################################

def deSkewText(binary,img):
    """
    Function to make the text aligned wrt horizontal axis

    @param binary: Binary form of the image with skewed/non-aligned text
    @param img: Original image having that text
    @returns: The image having text properly aligned to the horizontal axis
    """

    # In case of black text on white background(most part white)
    # invert the binary image
    if np.mean(binary) > 127:
        binary = np.bitwise_not(binary)

    contours,_ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    textContours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    coords = np.vstack(textContours).squeeze()
    rect = cv.minAreaRect(coords)
    points = cv.boxPoints(rect)

    cv.drawContours(img,[points.astype(int)],-1,(255,0,0),2)
    # cv.imshow("Rect",img)
    angle = rect[-1]

    if angle > 45:
        angle = -(90 - angle)
        
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(binary, M, (w, h),
	flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

#######################################################################################

def hasText(img):
    """
    Function to check the possiblity of text in the passed image

    @param img: The image with the possible text
    @returns: A boolean telling whether the image has some text
    """

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

    avg = np.mean(closed)
    return avg >= 75

#######################################################################################

def extractNumberPlate(gray,img):
    """
    Main Function to extract the number plate text from all the vehicles' number plate
    present in the frame

    @param gray: Gray form of image/frame
    @param img: Original image having vehicles/ number plates
    @returns: List of number plate texts in string format
    """

    #Blurring to reduce the noise
    #That will get rid of some minor edges
    ksize = (5,5)
    # blur = cv.bilateralFilter(gray, 13, 15, 15)
    blur = cv.GaussianBlur(gray,ksize,0)
    edged = cv.Canny(blur, 30, 200) #Edge detection
    # cv.imshow("Edged",edged)
    
    #Finding contours and keeping only the top 20 based on the area
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
            # winName = "Box {}".format(count)
            # cv.imshow(winName,recImg)
            
            gray = cv.cvtColor(recImg,cv.COLOR_BGR2GRAY)
            
            ret,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

            # Deskew the image text to make it straight
            # That would avoid unneccessary errors during the OCR stage
            fixed = deSkewText(thresh,recImg)
            # winName = "Result {}".format(count)
            # cv.imshow(winName,fixed)
            
            # Reading the text from the image
            text = readImage(fixed,"eng")
            npText.append(text)
            count += 1
            # print("Read some plates!")
        
    return npText  

#######################################################################################
##############################  Driver Codes  ##########################################
#######################################################################################
 
# Image testing driver code
# img = cv.imread("Photos/front/cars1.jpg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Original",img)
# plates = extractNumberPlate(gray,img)
# for plate in plates:
#     print(plate)

# Video testing driver code

# Uncomment when sending the plates to the firebase
# initFirebase() 

cap = cv.VideoCapture(RESOURCE_PATH)
cap.set(10,CAPTURE_BRIGHTNESS)

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
            # cv.imshow("Largest Moving",croppedFrame)
            plates = extractNumberPlate(croppedGray,croppedFrame)
        
            for plate in plates: 
                if validPlate(plate):
                    print(plate) 

                    # Uncomment when sending plates data to the firebase db
                    # postPlate(plate) 
                    pass
           
        frameNumber = 0
     
    if success: cv.imshow("Cam Video",frame)
    q = cv.waitKey(1)
    if q == ord('q'):
        break

cv.waitKey(0)
print("Terminating the license plate detection")
cap.release()
cv.destroyAllWindows()