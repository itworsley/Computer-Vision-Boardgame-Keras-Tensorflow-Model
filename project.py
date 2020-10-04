"""
COSC428 Assigment 2020
Author: Isaac Worsley
Date: 4/10/2020
"""

import os
import cv2
import math
import collections
import numpy as np
from PIL import Image
import time
from train import predict
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import time


# Whether or not to show popup windows containing preprocessing video frames.
SHOW_ALL_FRAMES = False

# Change if wanting to save training images.
SAVE_TRAINING_IMAGES = False

# Determine if writing to the test folder when saving images (save_image).
WRITE_TO_TEST_FOLDER = False

# Image dimensions
IMAGE_WIDTH, IMAGE_HEIGHT = 60, 60

# Set up the model to pass into the prediction network.
MODEL_VGG = applications.VGG16(include_top=False, weights='imagenet')

def main():
    """
    The main function of the script. Runs a while loop while video is available
    to continuously evaluate the live feed.    
    """
    # You may need to change the 0 to -1 depending on your camera set up.
    video = cv2.VideoCapture(0) 
    
    start_time = time.time()
    x = 1 # displays the frame rate every 1 second
    counter = 0    
    
    # Loop video display.
    while(video.isOpened()):
        ret, frame = video.read()
        copied = frame.copy()
        blurred = cv2.GaussianBlur(frame, (9,9), 0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Display original, blurred and grey video feed.
        cv2.imshow("Original", frame)
        cv2.moveWindow("Original", 100, 100)
        
        if SHOW_ALL_FRAMES:
            cv2.imshow("Blurred", blurred)
            cv2.imshow("Grey", grey)
            cv2.moveWindow("Blurred", 800, 100)
            cv2.moveWindow("Grey", 1500, 100)
        
        get_circles(video, frame, grey) 
        
        # Close the script when q is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break 
        
        counter+=1
        if (time.time() - start_time) > x :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()        
          
    # Release artifacts.
    video.release()
    cv2.destroyAllWindows()

def get_circles(video, frame, preProcessedVideo):
    """
    Determine the HoughCircles within the `preProcessedVideo`. Creates a mask of
    same dimensions as the original video, and places the calculated circles on
    the mask.
    
    Parameters
    ----------
    video : numpy.ndarray
            The live video feed.
    frame : numpy.ndarray
            The current video frame.
    preProcessedVideo: numpy.ndarray
                       The preprocessed video feed that is run through 
                       GassianBlur and greyscale.
    """
    
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  
    mask = np.zeros((int(height),int(width)), np.uint8)
    
    circles = cv2.HoughCircles(preProcessedVideo, cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                        param2=40, minRadius=20, maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Draw the circles in the mask image.
            cv2.circle(mask,(i[0],i[1]),(i[2]-5),(255,255,255),-1)
        
        extract_circles(mask, frame)

def extract_circles(mask, frame):
    """
    Determines the countours within the given frame and mask. Then iterates over
    these contours to find the bounding rectangle for each character. Uses the 
    predict method to determine the text within the given bounding rectangle and
    displays the current frame, with the suggested token highlighted on screen.
    
    Parameters
    ----------
    mask : numpy.ndarray
           Contains the inverse positioning of the circles within the frame.
    frame : numpy.ndarray
            The current video frame.             
    """
    # Join mask and original frame.
    masked_data = cv2.bitwise_and(frame, frame, mask=mask)

    
    if SHOW_ALL_FRAMES:
        cv2.imshow("Joined", masked_data)
        cv2.moveWindow("Joined", 100, 800) 

    # Find contours.
    contours, _ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    tokens = dict()
    
    for i, contour in enumerate(contours, 0):
        
        x,y,w,h = cv2.boundingRect(contour)
    
        # Crop masked_data.
        crop = masked_data[y:y+h,x:x+w]
        
        image_width = crop.shape[1]
        image_height = crop.shape[0]
        square_side_length = (image_width / (math.sqrt(2)))
        x1 = (image_width - square_side_length) / 2
        y1 = (image_height - square_side_length) / 2
        x2 = (image_width + square_side_length) / 2
        y2 = (image_height + square_side_length) / 2
        crop_img = crop[int(y1):int(y2), int(x1):int(x2)]               
        
        # Resize the image to given width and height used to train the model.
        resized_img = cv2.resize(crop_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        
        if SAVE_TRAINING_IMAGES: save_image(crop_img, "r")
        
        
        # Predict what is currently in the image.
        imageText = predict(resized_img, MODEL_VGG)
        
            
        tokens[imageText] = [x, y, w, h, image_width]
    
    targetLetter = get_target_letter(tokens)
    value = tokens[targetLetter]     
    
    # Draw a circle around the suggested target.
    cv2.circle(frame,(int(value[0]+(value[2]/2)),int(value[1]+(value[3]/2))),int(value[4] / 2),(0,255,0),4)
    
    ## Display suggested target token.    
    cv2.imshow("Suggested", frame)


def validate_images():
    """
    Used to check that the images in the validate directory are detected 
    correctly. Results are written to a .txt file.
    """
    file1 = open("validate/letters.txt","a") 
    
    for i in range(0, 18):
        original_img = cv2.imread("validate/{}.jpg".format(i))
        resized_img = cv2.resize(original_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite("validate/{}_resize.jpg".format(i), resized_img)
        file1.write(predict(resized_img, MODEL_VGG) + "\n")
    
    file1.close()       

def get_target_letter(tokens):
    """
    Determines the target letter based on the given target letters.
    
    Parameters
    ----------
    tokens : dict
             Keys are alphabetical letters, values are positions of circle. 
             
    Returns
    -------
    targetLetter : string
                   Given the tokens, the letter that is to be highlighted.
    """
    targetLetter = "z"
    
    for key, value in tokens.items():
        if (key < targetLetter):
            targetLetter = key    
    
    return targetLetter
    
    
def save_image(img, folderName):
    """
    Creates an images based on the target image to train the neural network algorithm. 
    350 of these images are placed in the trainDataFolder, every 10th image is placed
    in the testDataFolder (resulting in 35 images). The remaining images are placed in
    a sampleDataFolder (if exists).
    
    Parameters
    ----------
    img : numpy.ndarray
          The image to be saved.
    folderName : string
                 The target directory identifier for the image.
    """
    
    global WRITE_TO_TEST_FOLDER
    trainDataFolder = "train_data/train/" + folderName
    testDataFolder = "train_data/test/" + folderName
    sampleDataFolder = "train_data/sample"
    if not os.path.exists(trainDataFolder):
        os.mkdir(trainDataFolder)
    if not os.path.exists(testDataFolder):
        os.mkdir(testDataFolder)    
        
    trainFolderSize = len(os.listdir(trainDataFolder))
    testFolderSize = len(os.listdir(testDataFolder))
    
    # Print out the size of the training folder, in order to determine if the
    # circles are being retrieved.
    print(trainFolderSize)
    
    if (WRITE_TO_TEST_FOLDER):
        cv2.imwrite("{}/{}.jpg".format(testDataFolder, time.time()), img)
        WRITE_TO_TEST_FOLDER = False
    
    elif (trainFolderSize < 350): 
        cv2.imwrite("{}/{}.jpg".format(trainDataFolder, time.time()), img)      
        if (trainFolderSize % 10 == 0 and trainFolderSize != 0 and not WRITE_TO_TEST_FOLDER):
            WRITE_TO_TEST_FOLDER = True
        else:
            WRITE_TO_TEST_FOLDER = False
    else:
        if (testFolderSize < 35):
            cv2.imwrite("{}/{}.jpg".format(testDataFolder, time.time()), img)          
        elif (trainFolderSize == 350):
            cv2.imwrite("{}/{}.jpg".format(sampleDataFolder, time.time()), img)  


if __name__ == "__main__":
    
    main()