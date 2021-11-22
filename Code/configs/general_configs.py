from pathlib import Path
import cv2
from matplotlib import pyplot as plt

DEBUG = False
DATA_DIR_PATH = Path('../Data')
TRAIN_DATA_DIR_PATH = DATA_DIR_PATH / 'antrax/train/10,000x - 41'
OUTPUT_DIR_PATH = Path('../Output')
CONFIGS_DIR_PATH = Path('./configs')

# PREPROCESSING CONFIGS
# - Determines the height of the bar (in pixels) which should be removed from each image
BAR_HEIGHT = 70  # pix

# CALLBACKS
LOG_INTERVAL = 10
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_FACTOR = 0.1
LR_REDUCE_PATIENCE = 10
LAYER_VIZ_FIG_SIZE = (25, 25)
LAYER_VIZ_CMAP = 'gray'

# AUGMENTATION CONFIGS
BRIGHTNESS_DELTA = 0.1
CONTRAST = (0.4, 0.6)


if __name__ == '__main__':
    image_file = TRAIN_DATA_PATH / 'FILE_NAME'
    img = cv2.imread(image_file)
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(img, cmap='gray')

    ax[1].imshow(img[:-BAR_HEIGHT], cmap='gray')
