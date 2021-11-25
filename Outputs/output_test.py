import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path

INPUT_DIR = pathlib.Path('C:/Users/mchls/Desktop/Projects/Nano-Scout/Data/Test Images - 2 classes/train/10,000x - 89')
INPUT_DIR.is_dir()
CURRENT_DIR = pathlib.Path('C:\\Users\\mchls\\Desktop\\Projects\\Nano-Scout\\Outputs')
IMAGES_DIR = CURRENT_DIR / '2 class (Antrax, Strep) 5-KNN images'
if not IMAGES_DIR.is_dir():
    os.makedirs(IMAGES_DIR)
priors_df = pd.read_pickle('C:\\Users\\mchls\\Desktop\\Projects\\Nano-Scout\\Outputs\\priors_knn_df.pkl')
priors_df
priors_df.loc[:, 'file'] = priors_df.loc[:, 'file'].apply(lambda file_path: file_path.split('/')[-1])
priors_df.loc[1, 'neighbors'][0][1]
priors_df.loc[:, 'neighbors'] = priors_df.loc[:, 'neighbors'].apply(lambda file_paths: [file_path.split('/')[-1] for file_path in file_paths[0]])
priors_df
priors_df
# for root, dirs, files in os.walk(INPUT_DIR):
for idx, file in enumerate(priors_df.loc[:, 'file']):
    fig, axs = plt.subplots(1, 5, figsize=(100, 15));
    neighbors = priors_df.loc[idx, 'neighbors']
    for idx, neighbor in enumerate(neighbors):
        axs[idx].imshow(cv2.imread(root + f'/{neighbor}'))
        if not idx:
            axs[idx].set(title=f'Original ({neighbor})')
        else:
            axs[idx].set(title=f'{neighbor}')
        axs[idx].title.set_size(70)
    fig.savefig(IMAGES_DIR / file)
    # break

