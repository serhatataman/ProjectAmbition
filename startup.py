from keras.layers import Input
from keras.models import Model
import keras.optimizer_v2.adam
import os
import numpy as np
from PIL import Image
import Discriminator.discriminator as discriminator_builder
import Generator.generator as generator_builder
import Configurations.configuration as config
import DataWashing.image_resizer as data


# save_images function takes to count and noise as an input
# It generates frames from the parameters we’ve defined above
# and stores our generated image array which are generated from the noise input. Then it saves it as an image
def save_images(cnt, noise):
    image_array = np.full((
        config.PREVIEW_MARGIN + (config.PREVIEW_ROWS * (config.IMAGE_SIZE + config.PREVIEW_MARGIN)),
        config.PREVIEW_MARGIN + (config.PREVIEW_COLS * (config.IMAGE_SIZE + config.PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)

    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0

    for row in range(config.PREVIEW_ROWS):
        for col in range(config.PREVIEW_COLS):
            r = row * (config.IMAGE_SIZE + config.PREVIEW_MARGIN) + config.PREVIEW_MARGIN
            c = col * (config.IMAGE_SIZE + config.PREVIEW_MARGIN) + config.PREVIEW_MARGIN
            image_array[r:r + config.IMAGE_SIZE, c:c + config.IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)


training_data = data.get_resized_images()


# We have defined our input shape: which is 128X128X3 (image_size, image_size, image_channel)
image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)

# We are calling our build_discriminator function
# and passing the image shape then compiling it with a loss function and an optimizer.
discriminator = discriminator_builder.build_discriminator(image_shape)
# Since it is a classification model, we are using accuracy as its performance metric.
discriminator.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
# We are calling our build_generator function and passing our random_input noise vector as its input
generator = generator_builder.build_generator(config.NOISE_SIZE, config.IMAGE_CHANNELS)

random_input = Input(shape=(config.NOISE_SIZE,))

generated_image = generator(random_input)

# One important part of GAN is we should prevent our discriminator from training.
# Since we are only training generators here, we do not want to adjust the weights of the discriminator.
# This is what really an “Adversarial” in Adversarial Network means.
# If we do not set this, the generator will get its weight adjusted so it gets better at fooling the discriminator
# and it also adjusts the weights of the discriminator to make it better at being fooled.
# We don’t want this. So, we have to train them separately and fight against each other.
discriminator.trainable = False

validity = discriminator(generated_image)

# We are then compiling the generative model with loss function and optimizer.
combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# We are defining two vectors as y_real and y_fake. These vectors are composed of random 0’s and 1’s values
y_real = np.ones((config.BATCH_SIZE, 1))
y_fake = np.zeros((config.BATCH_SIZE, 1))

# We are creating a fixed_noise: this will result in generating images that are saved later on
# which we can see it getting better on every iteration.
fixed_noise = np.random.normal(0, 1, (config.PREVIEW_ROWS * config.PREVIEW_COLS, config.NOISE_SIZE))

# We are going to iterate over our training data with the range of epochs we’ve defined
cnt = 1
for epoch in range(config.EPOCHS):
    idx = np.random.randint(0, training_data.shape[0], config.BATCH_SIZE)
    # During the iteration process, we are taking a sample from a real image and putting that on x_real.
    # After that, we are defining a noise vector
    # and passing that to our generator model to generate a fake image in x_fake.
    x_real = training_data[idx]

    noise = np.random.normal(0, 1, (config.BATCH_SIZE, config.NOISE_SIZE))
    x_fake = generator.predict(noise)

    # Then we are training our discriminator model in both real and fake images separately
    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

    # After training, we are taking the metric from both models and taking the average.
    # This way we get the metric for the discriminator model
    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

    # Now for the generator model, we are training it on our noise vector and y_real: which is a vector of 1’s
    # Here we are trying to train the generator. Overtime generator will get better from these inputs
    # and the discriminator will not be able to discriminate whether the input is fake or real
    generator_metric = combined.train_on_batch(noise, y_real)

    # Now in the end we have an if statement which checks for our checkpoint.
    # If it reaches the checkpoint then it saves the current iteration noise
    # and prints the current accuracy of generator and discriminator.
    if epoch % config.SAVE_FREQ == 0:
        save_images(cnt, fixed_noise)
        cnt += 1

    print(f"{epoch} epoch, Discriminator accuracy: {100 * discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}")

# One thing to note here, our combined model is based on the generator model linked directly to the discriminator model.
# Here our Input is what the generator wants as an input: which is noise and output is what the discriminator gives us.
