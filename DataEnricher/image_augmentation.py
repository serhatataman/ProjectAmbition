# Although our data sometimes may seem like a lot, it might not really be enough to properly train a GAN.
# If there isn’t enough variety of images, the GAN may overfit the model which will yield poor results,
# or, worse yet, fall into the dreaded state of “model collapse”, which will yield nearly identical images.

# StyleGAN2 has a built-in feature to randomly mirror the source images left-to-right.
# So this will effectively double the number of sample images. This is better, but still not great.
# A technique called Image Augmentation to increase the number of images by a factor of 7.
# The augmentation uses random rotation, scaling, cropping, and mild color correction
# to create more variety in the image samples.

import os
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

# set up the file paths
from_path = 'Dataset'
to_path = 'Result'

# set up some parameters
size = 1024
num_augmentations = 6

# set up the image augmenter
seq = iaa.Sequential([
    iaa.Rot90((0, 3)),
    # iaa.Fliplr(0.5),
    iaa.PerspectiveTransform(scale=(0.0, 0.05), mode='replicate'),
    iaa.AddToHueAndSaturation((-20, 20))
])

# loop through the images, resizing and augmenting
path, dirs, files = next(os.walk(from_path))
for file in sorted(files):
    print("Processing file: " + file)
    if not os.path.isdir(path):
        raise FileNotFoundError("Path " + path + " not found!")
    image = Image.open(path + "/" + file)
    if image.mode == "RGB":
        if not os.path.isdir(to_path):
            os.mkdir(to_path)
        image.save(to_path + "/" + file)
        image_resized = image.resize((size, size), resample=Image.BILINEAR)
        image_np = np.array(image_resized)
        images = [image_np] * num_augmentations
        images_aug = seq(images=images)
        for i in range(0, num_augmentations):
            im = Image.fromarray(np.uint8(images_aug[i]))
            to_file = to_path + "/" + file[:-4] + '_' + str(i).zfill(2) + '.jpg'
            im.save(to_file)  # , quality=95)
