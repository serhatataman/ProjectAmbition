# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = '../Dataset'
RESIZED_IMAGES_DIR = 'ResizedImages'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print("Resizing training images in directory '" + IMAGE_DIR + "'...")

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

# Creating the resized images directory
resized_images_file_path = RESIZED_IMAGES_DIR + "/resized_images_data.npy"
if os.path.isdir(RESIZED_IMAGES_DIR):
    if os.path.isfile(resized_images_file_path):
        os.remove(resized_images_file_path)
else:
    print("Removing the old resized images...")
    os.mkdir(RESIZED_IMAGES_DIR)

print('Saving training images to a file...')
np.save(resized_images_file_path, training_data)
