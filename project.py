"""
COSC428 Assigment 2020
Author: Isaac Worsley
Status: Development
"""

import cv2
import math
import numpy as np
from pytesseract import *
from PIL import Image

ORIGINAL_IMAGE_FOLDER = "images/original"

pytesseract.tesseract_cmd = r'C:\Users\Isaac\Downloads\jTessBoxEditor-2.2.0\jTessBoxEditor\tesseract-ocr\tesseract.exe'

def getCircles():
    original_img = cv2.imread("images/tokens3.jpg")
    original_img1 = cv2.imread("images/tokens3.jpg",0)
    blurred_image = cv2.GaussianBlur(original_img, (9,9), 0)
    
    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("{}/blurred.jpg".format(ORIGINAL_IMAGE_FOLDER), blurred_image)
    cv2.imwrite("{}/gray.jpg".format(ORIGINAL_IMAGE_FOLDER), gray_image)
    
    ## Create mask
    height,width = original_img1.shape
    
    mask = np.zeros((height,width), np.uint8)    
    loop = True

    while loop:
        
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                    param2=20, minRadius=110, maxRadius=120)    
        print(circles)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
    
            for i in circles[0,:]:
                # Draw the circles in the mask image.
                cv2.circle(mask,(i[0],i[1]),(i[2]-5),(255,255,255),-1)
                #cv2.imshow("ORIGINAL", original_img)
    
        
        masked_data = cv2.bitwise_and(original_img, original_img, mask=mask)
        
        # Apply threshold
        _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        
        # Find contours.
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)
        
        # Iterate through list of contours.
        
        for i, contour in enumerate(contours, 0):
            
            x,y,w,h = cv2.boundingRect(contour)
        
            # Crop masked_data
            crop = masked_data[y:y+h,x:x+w]
            
            image_width = crop.shape[1]
            image_height = crop.shape[0]
            square_side_length = (image_width / (math.sqrt(2))- 10)
            x1 = (image_width - square_side_length) / 2
            y1 = (image_height - square_side_length) / 2
            x2 = (image_width + square_side_length) / 2
            y2 = (image_height + square_side_length) / 2
            crop_img = crop[int(y1):int(y2), int(x1):int(x2)]
            
            #cv2.rectangle(original_img, (x+40,y+40), (x+(w-40),y+(h-40)), (0, 255, 0), thickness=1)
            cv2.imwrite("images/{}.jpg".format(i), crop_img)
            
            processOneImage(i)

            if (i == (len(contours) - 1)):
                loop = False
        #cv2.imshow("Cropped Image".format(i), original_img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            loop = False        
    

def processOneImage(imageName):
    kernel = np.ones((5, 5), np.uint8) 
    original_img = cv2.imread("images/{}.jpg".format(imageName))
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9), 0)
    _,th2 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morphed = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    inverted = cv2.bitwise_not(morphed)
    outName = "{}-out".format(str(imageName))
    cv2.imwrite("images/{}.jpg".format(outName), inverted)
    
    # Whitelist letters in the alphabet.
    #--psm 10 means recognise 1 character.
    text = pytesseract.image_to_string(Image.open("images/{}.jpg".format(outName)), config=("-c tessedit"
                  "_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                  " --psm 10"
                  " -l engx+engxx"
                  " "))  
    print(outName  + " " + text)

#processOneImage("0")
getCircles()