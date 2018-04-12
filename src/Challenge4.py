# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

stepsPerEpoch = 8000

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (6, 6), input_shape = (128, 128, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (6, 6), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 12, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/rajeevranjan/challenge3Imgs/train',
target_size = (128, 128),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/home/rajeevranjan/challenge3Imgs/validation',
target_size = (128, 128),
batch_size = 32,
class_mode = 'categorical')
classifier.fit_generator(training_set,
steps_per_epoch = 60,
epochs = 5,
validation_data = test_set,
validation_steps = 60)
# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/rajeevranjan/axes/898743.jpeg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result[0][0])
print(result)
