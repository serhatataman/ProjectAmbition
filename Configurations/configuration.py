# Configuration

# Preview image Frame
# PREVIEW_ROWS and PREVIEW_COLS needs to be set to 1 if we want to create a single image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
# Margin should be set to zero in order to eliminate the white spaces around the image
PREVIEW_MARGIN = 4
SAVE_FREQ = 100

# Size vector to generate images from.
# NOISE_SIZE here is a latent dimension size to generate our images.
NOISE_SIZE = 100

# Configurations - Note: Images should always be of square size

# EPOCHS is a number of iterations: it defines how many times we want to iterate over our training images
# EPOCHS = 10000 -> original value
EPOCHS = 350
# BATCH_SIZE is a number of images to feed in every iteration.
BATCH_SIZE = 32
GENERATE_RES = 3
# IMAGE_SIZE is our image size which we resized earlier to 128X128
IMAGE_SIZE = 128  # rows/cols
# IMAGE_CHANNELS is a number of channel in our images; which is 3
IMAGE_CHANNELS = 3
