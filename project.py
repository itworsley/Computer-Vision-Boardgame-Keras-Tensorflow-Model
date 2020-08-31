"""
COSC428 Assigment 2020
Author: Isaac Worsley
Status: Development
"""

import cv2
import numpy as np

def getCircles():
    original_img = cv2.imread("tokens2.jpg")
    original_img1 = cv2.imread("tokens2.jpg",0)
    blurred_image = cv2.GaussianBlur(original_img, (9,9), 0)
    
    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)    
    
    ## Create mask
    height,width = original_img1.shape
    
    mask = np.zeros((height,width), np.uint8)    
    
    while True:
        
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                    param2=20, minRadius=0, maxRadius=100)        
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
    
            for i in circles[0,:]:
                # Draw the outer circle
                cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
    
        
        masked_data = cv2.bitwise_and(original_img, original_img, mask=mask)
        
        # Apply Threshold
        _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        
        # Find Contour
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)
        for i, contour in enumerate(contours, 0):
            
            x,y,w,h = cv2.boundingRect(contour)
        
            # Crop masked_data
            crop = masked_data[y:y+h,x:x+w]
            
            cv2.namedWindow("{}".format(i))
            cv2.moveWindow("{}".format(i), (i*100), (i*80))
            cv2.imshow("{}".format(i),crop)      
            cv2.imwrite("{}.jpg".format(i), crop)
        
        #x,y,w,h = cv2.boundingRect(np.array(contours[1]))
        
        ## Crop masked_data
        #crop2 = masked_data[y:y+h,x:x+w]
        
        #cv2.namedWindow('Second Image', cv2.WINDOW_NORMAL)
        #cv2.moveWindow('Second Image', 0, 500)
        #cv2.imshow('Second Image',crop2)     
        #cv2.imwrite("Second Image.jpg", crop2)
        
        #cv2.imshow("MASKED", mask)
        #cv2.imshow("BLURRED", blurred_image)
        #cv2.imshow("GRAY", gray_image)        
    
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break        
    

getCircles()