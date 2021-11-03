import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance

def get_image_file_pathes(root_dir_path):
    image_file_pathes = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            image_file_pathes.append(str(Path(root) / file))
    return image_file_pathes

class MorphDS(Sequence):
    def __init__(self, image_pathes: list, batch_size: int, central_crop_prop: float = 1.0, shuffle: bool = True):
        self.image_pathes = image_pathes
        self.batch_size = batch_size
        self.central_crop_prop = central_crop_prop if isinstance(central_crop_prop, float) and 0.0 < central_crop_prop  < 1.0 else 1.0
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_pathes) / self.batch_size))

    def __getitem__(self, index):
        # - Get the indices of the data in the range of current index
        batch_image_pathes = self.image_pathes[index * self.batch_size : (index + 1) * self.batch_size]

        return self.__data_generation(batch_image_pathes)

    def __data_generation(self, image_pathes):
        X = []
        for image_path in image_pathes:

            # - Crop the image in the center and normalize it
            img = tf.image.central_crop(cv2.imread(image_path), self.central_crop_prop) / 255

            # - Add the image to the batch list
            X.append(img)

        X = np.array(X, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(X)

        return X

    def on_epoch_end(self):
        pass

if __name__=='__main__':
    x=np.arange(10)
    np.random.shuffle(x)
    x
    data_root_dir_path = Path('D:\\Projects\\NanoScout\\code\\SCAN\\nn\\data')
    morph_ds = MorphDS(image_pathes = get_image_file_pathes(root_dir_path=data_root_dir_path), batch_size = 32, central_crop_prop = .7, shuffle = True)
    btch = next(iter(morph_ds))
    plt.imshow(btch[10])
    img_path = 'D:\\Projects\\NanoScout\\code\\SCAN\\nn\\data\\train\\90,000x - 1\\ANTHRAX R2 smp10009.tiff'
    img = tf.image.central_crop(cv2.imread(img_path), 0.75)
    img = tf.image.random_crop(cv2.imread(img_path), [32, 32, 1])
    type(img)
    plt.imshow(img)

    os.chdir(root_path)
    glob.glob('*.tiff')

    image_file_pathes

    l = ['a', 'b', 'c', 'd', 'e']
    append(['1', '2'])
    l
    i = np.random.choice(l, int(11*0.))
    k = np.setdiff1d(l, i)
    k

    a = np.arange(10)
    b = np.arange(10)
    c = np.arange(10)[::-1]
    distance.euclidean(a, c)
    np.linalg.norm(a - c)

    np.dot(a, b)
    np.dot(a, c)
