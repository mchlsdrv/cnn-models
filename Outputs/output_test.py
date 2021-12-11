import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path

INPUT_DIR = pathlib.Path('C:/Users/Michael/Desktop/Projects/Nano-Scout/Data/Test Images - 2 classes/train/10,000x - 89')
INPUT_DIR.is_dir()
CURRENT_DIR = pathlib.Path('C:/Users/Michael/Desktop/Projects/Nano-Scout/Outputs')
IMAGES_DIR = CURRENT_DIR / '2 class (Antrax, Strep) 5-KNN images'
if not IMAGES_DIR.is_dir():
    os.makedirs(IMAGES_DIR)
priors_df = pd.read_pickle(CURRENT_DIR / 'priors_knn_df.pkl')
priors_df
priors_df.loc[:, 'file'] = priors_df.loc[:, 'file'].apply(lambda file_path: file_path.split('/')[-1])
priors_df.loc[1, 'neighbors'][0][1]
priors_df.loc[:, 'neighbors'] = priors_df.loc[:, 'neighbors'].apply(lambda file_paths: [file_path.split('/')[-1] for file_path in file_paths[0]])
priors_df
priors_df
# for root, dirs, files in os.walk(INPUT_DIR):
len(priors_df.loc[0, 'distances'])
priors_df.shape
len(priors_df.loc[0, 'neighbors'])
for idx, file in enumerate(priors_df.loc[:, 'file']):
    fig, axs = plt.subplots(1, 5, figsize=(100, 15), facecolor='#c0d6e4');
    distances = priors_df.loc[idx, 'distances'][0]
    neighbors = priors_df.loc[idx, 'neighbors']
    for idx, (distance, neighbor) in enumerate(zip(distances, neighbors)):
        axs[idx].imshow(cv2.imread(str(INPUT_DIR / f'{neighbor}')))
        if not idx:
            axs[idx].set(title=f'{neighbor} (Original)')
        else:
            axs[idx].set(title=f'{neighbor} (Distance = {distance:.1f})')
        axs[idx].title.set_size(70)
    fig.savefig(IMAGES_DIR / file)
<<<<<<< HEAD
    # break
=======
    plt.close(fig)

fig, ax = plt.subplots(1, 2, facecolor='#c0d6e4')
ax[0].plot([0, 1, 2], [0, 1, 2])
ax[1].plot([0, 1, 2], [0, 1, 2])
fig.savefig('new_plot.png')
<<<<<<< HEAD

a = np.arange(10)
tf.convert_to_tensor(a, dtype=tf.float32)
a  = bytes('hello', 'utf-8')

type(a)
if isinstance(a, bytes):
    print('hell')
    
=======
>>>>>>> e621c3d326960ac7bd7d902114b3d152b2def4c8
>>>>>>> c2b420f6fc53c227fbca01e1b61be11351f7e2a2
