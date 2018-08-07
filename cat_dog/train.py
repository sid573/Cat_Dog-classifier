import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.callbacks import History



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test',target_size = (64, 64),batch_size = 32,class_mode = 'binary')

history = model.fit_generator(training_set,epochs = 50,validation_data = test_set,validation_steps = 2000)

my_model_json = model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(my_model_json)
# serialize weights to HDF5
model.save_weights("model_weight.h5")

model.save('my_model.h5')
