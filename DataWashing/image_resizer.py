import os
import numpy as np
from PIL import Image
import Configurations.configuration as config
import startup

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_DIR = '../Dataset'
RESIZED_IMAGES_DIR = 'ResizedImages'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print("Resizing training images in directory '" + IMAGE_DIR + "'...")

# We are using Pillow to resize all images to our desired size and appending them on a list as numpy array

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.ANTIALIAS)

    startup.training_data.append(np.asarray(image))

# We are using numpy to reshape the array in a suitable format and normalizing data
training_data = np.reshape(startup.training_data, (-1, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

# Creating the resized images directory
resized_images_file_path = RESIZED_IMAGES_DIR + "/resized_images_data.npy"
if os.path.isdir(RESIZED_IMAGES_DIR):
    if os.path.isfile(resized_images_file_path):
        os.remove(resized_images_file_path)
else:
    print("Removing the old resized images...")
    os.mkdir(RESIZED_IMAGES_DIR)

# We are saving our image array in npy binary file so that we don’t have to go through all the images every time
print('Saving training images to a file...')
np.save(resized_images_file_path, training_data)
