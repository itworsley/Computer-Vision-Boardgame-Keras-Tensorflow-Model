# importing libraries
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
  
  
img_width, img_height = 60, 60

train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/test'
sample_data_dir = 'train_data/sample'
deleted_data_dir = 'train_data/deleted'

num_classes = 4

def resizeImages():
    currentDirectory = sample_data_dir
    for i, filename in enumerate(os.listdir(currentDirectory)):
        img = cv2.imread("{}/{}".format(currentDirectory,filename))
        #os.remove("{}/{}".format(currentDirectory,filename))
        res = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite("{}/{}.jpg".format(currentDirectory, i+1), res)
        
    print("\n" + "*"*50)
    print("Rewrote files in {}".format(currentDirectory))
  
def trainKeras():  
    nb_train_samples = 1050 
    nb_validation_samples = 90
    epochs = 20
    batch_size = 16
    
      
    model = createModel()
      
    model.compile(loss ='binary_crossentropy', 
                         optimizer ='rmsprop', 
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
      
    model.save_weights('keras_model.h5') 
    
    
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
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes)) 
    model.add(Activation('softmax')) 
    return model

def loadSamples():
    images = []
    for file in os.listdir(sample_data_dir):
        img = image.load_img('{}/{}'.format(sample_data_dir, file), target_size=(img_width, img_height))
        #plt.imshow(img)
        #plt.show()
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    return images
    

def predictKeras():
    model = createModel()
    model.load_weights('keras_model.h5')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])    

    #img = image.load_img('train_data/test/a/1.jpg', target_size=(img_width, img_height))
    #a = image.img_to_array(img)
    #a = np.expand_dims(a, axis=0)
    
    #img = image.load_img('train_data/test/c/1.jpg', target_size=(img_width, img_height))
    #b = image.img_to_array(img)
    #b = np.expand_dims(b, axis=0)    
    
    images = loadSamples()
    #images = np.asarray(images)
    #print(len(images))
    
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=10)
    print(classes)
    #class_names = ["a", "b", "c", "d"]
    #train_labels = to_categorical(train_labels, num_classes)
    #print(train_labels)
    
    #predictions = model.predict(images)
    #print(predictions)
    
    #classes = np.argmax(predictions, axis = 1)
    #print(classes)    
    
    
    #classesValues = dict()
    #classesValues["0"] = "a"
    #classesValues["1"] = "c"
    
    ## Print the classes.
    #for val in classes:
        #print(classesValues[str(val[0])])
    
#trainKeras()
#predictKeras()
#resizeImages()