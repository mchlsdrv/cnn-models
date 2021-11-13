import os
import io
import threading
import tensorflow as tf
import matplotlib.pyplot as plt


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


def launch_tb(log_dir):
    os.system(f'tensorboard --logdir={log_dir}')

    th = threading.Thread(
        target=lambda logdir: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    th.start()
