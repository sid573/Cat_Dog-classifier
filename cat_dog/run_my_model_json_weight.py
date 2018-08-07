import os
import sys
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import History

fn = sys.argv[1]

json_file = open('my_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weight.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


test_image = image.load_img(fn, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print (result)

image = Image.open(fn)
image.show()