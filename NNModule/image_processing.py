import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config
from utils import create_dir_if_not_exists, delete_all_files_from, copy_file
from tqdm.auto import tqdm
import tensorflow as tf


def convert_to_grayscale(rgba):
    return np.where(rgba[:, :, -1] > 0., 1., 0.)


def open_rgba(path):
    original = plt.imread(path)
    original = np.asarray(original)

    return convert_to_grayscale(original)


def combine_images(src_dir: Path, destination_dir: Path, prefix='scene', channels=2+config.NUM_OF_ANGLES, starting_image_idx=0):
    assert(starting_image_idx + channels <= 2 + config.NUM_OF_ANGLES)

    print(f'Combining images from {src_dir} to {destination_dir}')
    image_paths = list(src_dir.glob('*.png'))
    padding = len(str(int(len(image_paths) / config.NUM_OF_ANGLES)))
    file_extension = '.npy' if channels > 3 else '.png'

    for i in tqdm(range(0, len(image_paths), 2 + config.NUM_OF_ANGLES)):
        combined = np.concatenate([open_rgba(image_paths[i + j + starting_image_idx])[..., np.newaxis] for j in range(channels)], axis=-1)
        output_fname = prefix + '_' + str(int(i / (2 + config.NUM_OF_ANGLES))).zfill(padding) + file_extension
        output_path = str(destination_dir / output_fname)

        if channels > 3:
            raw = tf.io.serialize_tensor(combined.astype(np.float16))
            tf.io.write_file(output_path, raw)
        else:
            plt.imsave(output_path, combined[..., :3])


if __name__ == "__main__":
    # Change the mode if you need
    dataset_generation = True

    if dataset_generation:
        create_dir_if_not_exists(config.DS_IMAGES)
        delete_all_files_from(config.DS_IMAGES)
        copy_file(config.DS_PROGRAMS_BLENDER, config.DS_PROGRAMS)
        combine_images(config.BLENDER_DIR, config.DS_IMAGES)
    else:
        combine_images(config.EVAL_TMP_DIR, config.EVAL_DIR, prefix='predicted', channels=3, starting_image_idx=0)
