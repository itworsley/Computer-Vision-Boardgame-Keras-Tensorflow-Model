# importing libraries
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.models import load_model
from keras.utils import to_categorical
from keras import applications
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import string
import shutil
  
model_name = "model.h5"  
img_width, img_height = 60, 60

train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/test'
sample_data_dir = 'train_data/sample'
deleted_data_dir = 'train_data/deleted'

num_classes = 4

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def resizeImages():
    currentDirectory = validation_data_dir + "/c"
    tempDirectory = currentDirectory + "/temp"
    if not os.path.exists(tempDirectory):
        os.mkdir(tempDirectory)
    
    numFilesReWrote = 0    
    for i, filename in enumerate(os.listdir(currentDirectory)):
        if (filename.endswith(".jpg")):
            img = cv2.imread("{}/{}".format(currentDirectory,filename))
            os.remove("{}/{}".format(currentDirectory,filename))
            res = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite("{}/{}.jpg".format(tempDirectory, i+1), res)
            numFilesRewrote += 1
    
    
    for file in os.listdir(tempDirectory):
        shutil.move("{}/{}".format(tempDirectory, file), currentDirectory)
        
    os.rmdir(tempDirectory)
    print("\n" + "*"*50)
    print("Rewrote {} files in {}".format(numFilesReWrote, currentDirectory))
  
def trainKeras():  
    nb_train_samples = num_classes * 350 
    nb_validation_samples = num_classes * 35
    epochs = 20
    batch_size = 16
    
      
    model = createModel()
      
    model.compile(loss ='categorical_crossentropy', 
                         optimizer ='adam', 
                       metrics =['accuracy']) 
      
    train_datagen = ImageDataGenerator( 
                    rescale = 1. / 255, 
                     shear_range = 0.2, 
                      zoom_range = 0.2, 
                horizontal_flip = True) 
      
    test_datagen = ImageDataGenerator(rescale = 1. / 255) 
      
    train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                  target_size =(img_width, img_height), 
                         batch_size = batch_size, class_mode ='categorical') 
      
    validation_generator = test_datagen.flow_from_directory( 
                                        validation_data_dir, 
                       target_size =(img_width, img_height), 
              batch_size = batch_size, class_mode ='categorical') 
      
    model.fit_generator(train_generator, 
        steps_per_epoch = nb_train_samples // batch_size, 
        epochs = epochs, validation_data = validation_generator, 
        validation_steps = nb_validation_samples // batch_size) 
      
    model.save_weights(model_name) 
    
    
def createModel():
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 
        
    model = Sequential() 
    
    
    model.add(Conv2D(32, (3, 3), input_shape = input_shape)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
      
    model.add(Conv2D(32, (3, 3))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
      
    model.add(Conv2D(64, (3, 3))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
      
    model.add(Flatten()) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes)) 
    model.add(Activation('softmax')) 
    
    ## Creating a Sequential model
    #model= Sequential()
    #model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=input_shape))
    #model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    #model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    #model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
    
    
    #model.add(Flatten())
    
    #model.add(Dense(20,activation='relu'))
    #model.add(Dense(15,activation='relu'))
    #model.add(Dense(num_classes,activation = 'softmax'))
    #model = Sequential()
    #model.add(Flatten(input_shape=input_shape))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(num_classes, activation='softmax'))    
    #print(model.summary())
    return model

def loadSamples():
    images = []
    for file in os.listdir(sample_data_dir):
        img = image.load_img('{}/{}'.format(sample_data_dir, file), target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        images.append(img)
    return images
    

def predictKeras():
    images = loadSamples()
    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_prediction = model.predict(images[0])
    print(bottleneck_prediction)
    
    model = createModel()
    
    model.load_weights(model_name)
    
    #model.compile(loss='categorical_crossentropy',
                  #optimizer='adam',
                  #metrics=['accuracy'])    
    
    
    ##images = np.vstack(images)
    
    classes = model.predict_classes(bottleneck_prediction)
    print(classes)
    probabilities = model.predict_proba(bottleneck_prediction)
    #alphabet = dict(zip(range(0,18), string.ascii_lowercase[:18]))
    alphabet = dict()
    alphabet[0] = "a"
    alphabet[1] = "b"
    alphabet[2] = "c"
    alphabet[3] = "d"
    
    # Print the classes.
    for val in classes:
        print(alphabet[int(val)])
    
#trainKeras()
predictKeras()
#resizeImages()