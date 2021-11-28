from pathlib import Path
import cv2
from matplotlib import pyplot as plt

DEBUG = False
DATA_DIR_PATH = Path('../Data')
TRAIN_DATA_DIR_PATH = DATA_DIR_PATH / 'Test Images - 2 classes/train/10,000x - 89'
OUTPUT_DIR_PATH = Path('../Output')
CONFIGS_DIR_PATH = Path('./configs')

# PREPROCESSING CONFIGS
# - Determines the height of the bar (in pixels) which should be removed from each image
BAR_HEIGHT = 70  # pix

# CALLBACKS
# - Early Stopping
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 0
EARLY_STOPPING_MODE = 'auto'
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_VERBOSE = 1

# - Tensor Board
TENSOR_BOARD_WRITE_GRAPH = True
TENSOR_BOARD_WRITE_IMAGES = True
TENSOR_BOARD_WRITE_STEPS_PER_SECOND = True
TENSOR_BOARD_UPDATE_FREQ = 'epoch'
TENSOR_BOARD_LOG_INTERVAL = 10

# - LR Reduce
LR_REDUCE_MONITOR = 'val_loss'
LR_REDUCE_FACTOR = 0.1
LR_REDUCE_PATIENCE = 100
LR_REDUCE_MIN_DELTA = 0.0001
LR_REDUCE_COOLDOWN = 0
LR_REDUCE_MIN_LR = 0.0
LR_REDUCE_MODE = 'auto'
LR_REDUCE_VERBOSE = 1

# - Layer Visualization
CONV_VIS_LAYER_FIG_SIZE = (25, 25)
CONV_VIS_LAYER_CMAP = 'gray'
CONV_VIS_LAYER_LOG_INTERVAL = 10

# - Model Checkpoint
MODEL_CHECKPOINT_VERBOSE = 1
MODEL_CHECKPOINT_CHECKPOINT_FREQUENCY = 500
MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY = True

# AUGMENTATION CONFIGS
BRIGHTNESS_DELTA = 0.1
CONTRAST = (0.4, 0.6)


if __name__ == '__main__':
    image_file = TRAIN_DATA_PATH / 'FILE_NAME'
    img = cv2.imread(image_file)
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(img, cmap='gray')

    ax[1].imshow(img[:-BAR_HEIGHT], cmap='gray')
