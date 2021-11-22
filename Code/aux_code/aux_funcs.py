import os
import io
import threading
import tensorflow as tf
import matplotlib.pyplot as plt


def get_patch_df(image_file, patch_height, patch_width):
    assert image_file.is_file(), f'No file \'{image_file}\' was found!'

    img = cv2.imread(str(image_file))
    df = pd.DataFrame(columns=['file', 'image'])
    img_h, img_w, _ = img.shape
    for h in range(0, img_h, patch_height):
        for w in range(0, img_w, patch_width):
            patch = img[h:h+patch_height, w:w+patch_width, :]
            df = df.append(dict(file=image_file, image=patch), ignore_index=True)
    return df


def transform_images(images_root_dir, model, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:
            df = df.append(get_patch_df(image_file=Path(f'{root}/{file}'), patch_height=patch_height, patch_width=patch_width), ignore_index=True)
    df.loc[:, 'vector'] = df.loc[:, 'image'].apply(lambda x: model(np.expand_dims(x, axis=0)) if len(x.shape) < 4 else model(x))
    return df


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def launch_tensor_board(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th
