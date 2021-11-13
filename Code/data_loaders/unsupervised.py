import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
import types


class MorphologicalDataSetLoader(Sequence):
    def __init__(self, image_files: list, preprocessing_func: types.FunctionType, batch_size: int, val_prop: float = 0.0, shuffle: bool = True):

        self.image_files = image_files
        self.preprocessing_func = preprocessing_func
        self.batch_size = batch_size
        self.val_prop = val_prop
        self.shuffle = shuffle
        self.val_image_files = np.random.choice(self.image_files, int(len(self.image_files) * self.val_prop))
        self.train_image_files = np.setdiff1d(self.image_files, self.val_image_files)

        print(f'''
        Total files: {len(self.image_files)}
        - Train files: {len(self.train_image_files)}
        - Validation files: {len(self.val_image_files)}
        ''')

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # - Get the indices of the data in the range of current index
        train_batch_image_files = self.train_image_files[index * self.batch_size : (index + 1) * self.batch_size]

        return self.load_images(train_batch_image_files)

    def load_images(self, image_files):
        X = []
        for image_path in image_files:
            # - Add the image to the batch list
            X.append(self.preprocessing_func(cv2.imread(image_path)))

        X = np.array(X, dtype=np.float32)
        if self.shuffle:
            np.random.shuffle(X)

        return X

    def get_val_data(self):
        return self.load_images(self.val_image_files)

    def on_epoch_end(self):
        pass
