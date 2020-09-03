import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
tf.get_logger().disabled = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# Image dimensions
img_width, img_height = 60, 60

top_model_weights_path = 'model.h5'
train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/test'
sample_data_dir = 'train_data/sample'
deleted_data_dir = 'train_data/deleted'

epochs = 30
batch_size = 16

def save_bottlebeck_features():
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)
    
    
def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    np.save('class_indices.npy', generator_top.class_indices)

    train_data = np.load('bottleneck_features_train.npy')

    train_labels = generator_top.classes

    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))


    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path) 
    
    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))    
    
    
def predictSamples():
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()

    num_classes = len(class_dictionary)

    images = loadSamples()
    results = []
    for image in images:
        model = applications.VGG16(include_top=False, weights='imagenet')
        bottleneck_prediction = model.predict(image)
    
        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))
    
        model.load_weights(top_model_weights_path)
    
        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)
    
        probabilities = model.predict_proba(bottleneck_prediction)
    
        inID = class_predicted[0]
    
        inv_map = {v: k for k, v in class_dictionary.items()}
    
        label = inv_map[inID]
        results.append(label)
    print(results)


def predict(image):
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy', allow_pickle=True).item()

    num_classes = len(class_dictionary)

    modelVGG = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_prediction = modelVGG.predict(image)
    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}
    
    return inv_map[inID]

def loadSamples():
    images = []
    for file in os.listdir(sample_data_dir):
        image = load_img('{}/{}'.format(sample_data_dir, file), target_size=(img_width, img_height))
        image = img_to_array(image)
        image = image / 255
        image = np.expand_dims(image, axis=0)
        images.append(image)
    return images


print(predict(load_img('{}/{}'.format(sample_data_dir, "2.jpg"), target_size=(img_width, img_height))))

cv2.destroyAllWindows()