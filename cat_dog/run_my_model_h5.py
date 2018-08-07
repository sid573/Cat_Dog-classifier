from keras import backend as K
import os
import sys
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt


fn = sys.argv[1]

loaded_model = load_model('my_model.h5')
##preds = predict(loaded_model, fn,target_size = (64, 64))

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


test_image = image.load_img(fn, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print (result)