import Configurations.configuration as config
from datetime import datetime
import os
import numpy as np
from PIL import Image


# save_images function takes to count and noise as an input
# It generates frames from the parameters we’ve defined above
# and stores our generated image array which are generated from the noise input. Then it saves it as an image
def image_saver(generator, noise):
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
    filename = os.path.join(output_path, f"trained-{image_count}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
