import cv2 as cv
import numpy as np

def deSkewText(binary,img):
    print(np.mean(binary))
    if np.mean(binary) > 127:
        binary = np.bitwise_not(binary)

    contours,_ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    textContours = sorted(contours, key=cv.contourArea, reverse=True)

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

img = cv.imread("Photos/skewed_text/test2.jpg")
cv.imshow("Text Image",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image",gray)

_,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Binary",thresh)

fixed = deSkewText(thresh,img)
cv.imshow("Fixed",fixed)
# contours,_ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img,contours,-1,(255,0,0),1)
# cv.imshow("Contours",img)

cv.waitKey(0)

