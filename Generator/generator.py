from keras.layers import Input, Reshape, Dense, BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import Configurations.configuration as config


# Generator model takes noise as an input and outputs an image
def build_generator(noise_size, channels):
    model = Sequential()

    # Note: From the shape of 4X4 it will be extended up to the size of 128X128 which is our training_data shape
    # Since our generator model has to generate images from noise vector,
    # our first layer is a fully connected dense layer of size 4096 (4 * 4 * 256) which takes noise_size as a parameter
    # Note: We have defined its size to be of 4096 to for resizing it in 4X4X256 shaped layer
    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    # We are using Reshape layer to reshape our fully connected layer in the shape of 4X4X256
    model.add(Reshape((4, 4, 256)))
    # Layer blocks after this are just a Convolutional layer with batch normalizations and activation function relu

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    for i in range(config.GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    input_noise = Input(shape=(noise_size,))
    generated_image = model(input_noise)

    return Model(input_noise, generated_image)
