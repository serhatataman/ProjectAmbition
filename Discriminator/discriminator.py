from keras.layers import Input, Dropout, Dense, Flatten, BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model


def build_discriminator(image_shape):
    # We are initializing a Sequential model from keras which helps us in creating linear stacks of layers
    model = Sequential()

    # Our first layer is a convolutional layer of 32 shape having
    # kernel_size of 3 and our stride value is 2 with padding same. Since it is a first layer it holds input_shape.
    # To understand what is going on here, you can refer to keras official documentation page
    # But in simple language, here we are defining a convolutional layer which has
    # a filter of size 3X3 and that filter strides over our image data.
    # We have padding of same which means, no additional paddings are added. It remains the same as the original.
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    # We are adding a LeakyRelu layer which is an activation function
    model.add(LeakyReLU(alpha=0.2))
    # Similarly in other block of layers are added in a sequential model
    # with some dropouts and batch normalization to prevent overfitting
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    # The last layer of our model is a Fully connected layer with an activation function sigmoid
    model.add(Flatten())
    # Since our discriminator’s job is to classify whether the given image is fake or not,
    # it is a binary classification task and sigmoid is an activation
    # that squeezes every value to values between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    
    input_image = Input(shape=image_shape)
    validity = model(input_image)

    return Model(input_image, validity)

# Discriminator in GAN uses a cross entropy loss, since discriminator's job is to classify;
# cross entropy is the best for classification
