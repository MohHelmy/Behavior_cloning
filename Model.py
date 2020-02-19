import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten,Dense,Cropping2D,Lambda,Conv2D,Dropout,Activation,MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
import math
from sklearn.utils import shuffle
from keras.utils import plot_model
import random
import os 
print (os.path.abspath(__file__))


#all training data
X_data = []
y_data = []






lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

 
images = []
images_left = []
images_right = []
measurements = []
for line  in lines:
    
    source_path = line[0]
    filename = source_path.split("\\")[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread (current_path)
    if image is None:
        print('center is None')
    images.append(image)
    measurement = float (line[3])
    measurements.append(measurement)
    
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)
    
    source_path = line[1]
    filename = source_path.split("\\")[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread (current_path)
    #if image is None:
        #print('left is None')
        #print(current_path)
        #print(filename)
    #images_left.append(image)
    images.append(image)
    
    calc_measurement =(measurement+random.uniform(0.01, 0.2))
    #calc_measurement =(measurement+.2)
    measurements.append(calc_measurement)

    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -calc_measurement
    measurements.append(measurement_flipped)
    
    source_path = line[2]
    filename = source_path.split("\\")[-1]
    current_path = "./data/IMG/" + filename
    image = cv2.imread (current_path)
    #if image is None:
        #print('right is None')
        #print(current_path)
        #print(filename)
    #images_right.append(image)
    images.append(image)
    calc_measurement =(measurement-random.uniform(0.01, 0.2))
    #calc_measurement =(measurement-.2)
    measurements.append(calc_measurement)
    
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -calc_measurement
    measurements.append(measurement_flipped)

    

    
X_train = np.asarray(images)
Y_train = np.asarray(measurements)

#sklearn.utils.shuffle(X_train, Y_train)

print('Shape')
print(current_path)

#If model predictions are poor on both the training and validation set (for example, mean squared error is high on both), then this is evidence of 
# underfitting. Possible solutions could be:
#1-increase the number of epochs add more convolutions to the network.


#When the model predicts well on the training set but poorly on the validation set (for example, low mean squared error for training set, high mean squared
#error for validation set), this is evidence of overfitting. If the model is overfitting, a few ideas could be to
#1-use dropout or pooling layers
#2-use fewer convolution or fewer fully connected layers collect more data or further augment the data set

model = Sequential()
#Lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images
model.add(Lambda(lambda x: (x / 255.0) -.5 , input_shape=(160,320,3)))

#crop unwanted top an bottom sections fo clear images
#74 rows pixels from the top of the image
#20 rows pixels from the bottom of the image
#60 columns of pixels from the left of the image
#60 columns of pixels from the right of the image
model.add(Cropping2D(cropping = ((74,25), (0,0)),input_shape=(160, 320, 3)))

#Normalization
# starts with five convolutional and maxpooling layers
#model.add(Conv2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


model.add(Conv2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


model.add (Flatten ())


# Next, five fully connected layers
model.add (Dense (120))
model.add (Dense (84))
model.add (Dense (1))

model.summary()


model.compile(loss = 'mse' , optimizer = 'adam')

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


model.fit (X_train , Y_train , validation_split = 0.2 , shuffle = True , nb_epoch = 7)
#model.fit( x = X_train, y = Y_train , batch_size=500 , epochs=7 , verbose=1, shuffle=True, validation_split=0.2)

model.save('model.h5')
exit()

#-------------------End of trial code -------------------
