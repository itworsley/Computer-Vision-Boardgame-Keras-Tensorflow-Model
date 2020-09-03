"""
COSC428 Assigment 2020
Author: Isaac Worsley
Status: Development
"""

import os
import cv2
import math
import collections
import numpy as np
#from pytesseract import *
from PIL import Image
import time
from train import predict


ORIGINAL_IMAGE_FOLDER = "images/original"

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
currentIndex = 0

#pytesseract.tesseract_cmd = r'C:\Users\Isaac\Downloads\jTessBoxEditor-2.2.0\jTessBoxEditor\tesseract-ocr\tesseract.exe'

writeToTestFolder = False

def main():
    video = cv2.VideoCapture(0) 
    count = 0
    
    # Loop video display.
    while(video.isOpened()):
        ret, frame = video.read()
        copied = frame.copy()
        blurred = cv2.GaussianBlur(frame, (9,9), 0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Display original, blurred and grey video feed.
        cv2.imshow("Original", frame)
        cv2.moveWindow("Original", 100, 100)
        #cv2.imshow("Blurred", blurred)
        #cv2.imshow("Grey", grey)
        #cv2.moveWindow("Blurred", 800, 100)
        #cv2.moveWindow("Grey", 1500, 100)
        
        #if count%30 == 0:
        getCircles(video, frame, grey)
            
        count += 1
           
        # Close the script when q is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break 
    # Release artifacts.
    video.release()
    cv2.destroyAllWindows()

def getCircles(video, frame, processedVideo):
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  
    mask = np.zeros((int(height),int(width)), np.uint8)
    
    circles = cv2.HoughCircles(processedVideo, cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                        param2=40, minRadius=20, maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Draw the a green circle around each identified circle
            #cv2.circle(frame,(i[0],i[1]),(i[2]-5),(0,255,0),1)
            
            # Draw the circles in the mask image.
            cv2.circle(mask,(i[0],i[1]),(i[2]-5),(255,255,255),-1)
            
        #cv2.imshow("Mask", mask)
        #cv2.imshow("Circles", frame)    
        #cv2.moveWindow("Mask", 100, 600)
        #cv2.moveWindow("Circles", 800, 600)
        
        extractCircles(circles, mask, frame)
        
               


def extractCircles(circles, mask, frame):
    #global alphabet, currentIndex
    masked_data = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Apply threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)     

    # Find contours.
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    tokens = dict()
    for i, contour in enumerate(contours, 0):
        
        x,y,w,h = cv2.boundingRect(contour)
    
        # Crop masked_data
        crop = masked_data[y:y+h,x:x+w]
        
        image_width = crop.shape[1]
        image_height = crop.shape[0]
        square_side_length = (image_width / (math.sqrt(2)))
        x1 = (image_width - square_side_length) / 2
        y1 = (image_height - square_side_length) / 2
        x2 = (image_width + square_side_length) / 2
        y2 = (image_height + square_side_length) / 2
        crop_img = crop[int(y1):int(y2), int(x1):int(x2)]
        
        resized_img = cv2.resize(crop_img, (60, 60), interpolation=cv2.INTER_AREA)
        
        imageText = predict(resized_img)
        #saveImage(crop_img, "d")
        tokens[imageText] = [x, y, w, h, image_width]
        
    minKey = "z"
    for key, value in tokens.items():
        if (key < minKey):
            minKey = key
    
    value = tokens[minKey]        
    cv2.circle(frame,(int(value[0]+(value[2]/2)),int(value[1]+(value[3]/2))),int(value[4] / 2),(0,0,255),4)
        
    cv2.imshow("Suggested", frame)

    
def saveImage(crop_img, folderName):
    global writeToTestFolder
    trainDataFolder = "train_data/train/" + folderName
    testDataFolder = "train_data/test/" + folderName
    sampleDataFolder = "train_data/sample"
    if not os.path.exists(trainDataFolder):
        os.mkdir(trainDataFolder)
    if not os.path.exists(testDataFolder):
        os.mkdir(testDataFolder)    
        
    trainFolderSize = len(os.listdir(trainDataFolder))
    
    print(trainFolderSize)
    
    if (writeToTestFolder):
        cv2.imwrite("{}/{}.jpg".format(testDataFolder, time.time()), crop_img)
        writeToTestFolder = False
    
    elif (trainFolderSize < 350): 
        cv2.imwrite("{}/{}.jpg".format(trainDataFolder, time.time()), crop_img)      
        if (trainFolderSize % 10 == 0 and trainFolderSize != 0 and not writeToTestFolder):
            writeToTestFolder = True
        else:
            writeToTestFolder = False
    else:
        if (trainFolderSize == 350):
            cv2.imwrite("{}/{}.jpg".format(sampleDataFolder, time.time()), crop_img)    


main()