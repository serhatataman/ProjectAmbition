from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100

# Size vector to generate images from.
# NOISE_SIZE here is a latent dimension size to generate our images.
NOISE_SIZE = 100

# Configurations - Note: Images should always be of square sizeo

# EPOCHS is a number of iterations: it defines how many times we want to iterate over our training images
EPOCHS = 10000
# BATCH_SIZE is a number of images to feed in every iteration.
BATCH_SIZE = 32
GENERATE_RES = 3
# IMAGE_SIZE is our image size which we resized earlier to 128X128
IMAGE_SIZE = 128  # rows/cols
# IMAGE_CHANNELS is a number of channel in our images; which is 3
IMAGE_CHANNELS = 3

# Loading resized images
resized_images_path = os.path.join("DataWashing/ResizedImages/", "resized_images_data.npy")
if not os.path.exists(resized_images_path):
    raise FileNotFoundError("No resized images are found at directory: " + resized_images_path)

training_data = np.load(resized_images_path)

