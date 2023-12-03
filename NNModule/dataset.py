import os

import tensorflow as tf
import matplotlib.pyplot as plt
import config
import json
from pathlib import Path
import numpy as np

def read_scene_file(fname):
    scenes = []
    angles = []
    with open(fname, 'r') as file:
        for line in file:
            angle, primitives = line.split(';')
            scenes += [json.loads(primitives)]
            angles += [json.loads(angle)]

    return scenes, angles


def get_scene_dataset(imgs_dir,
                      scenes_filename,
                      batch_size=None,
                      validation_split=0.1,
                      shuffle=True,
                      only_slice=None):

    suffix = Path(os.listdir(config.DS_IMAGES)[0]).suffix
    image_ds = tf.data.Dataset.list_files(str(imgs_dir/('*' + suffix)), shuffle=False)
    if suffix == '.npy':
        image_ds = image_ds.map(lambda filename: tf.io.parse_tensor(tf.io.read_file(filename), tf.float16))
    else:
        image_ds = image_ds.map(lambda filename: tf.io.decode_png(tf.io.read_file(filename), channels=3) / 255)

    scenes, angles = read_scene_file(scenes_filename)
    scenes_ds = tf.data.Dataset.from_tensor_slices(scenes)
    angles_ds = tf.data.Dataset.from_tensor_slices(angles)

    # Normalizing angles (from [0, 2PI] to [0, 1])
    angles_ds = angles_ds.map(lambda angle: angle / (np.pi * 2.))

    # Normalizing scenes (everything to [0, 1])
    # [cidx, +/-, type, scale, rot, trans]
    scenes_ds = scenes_ds.map(
        lambda scene: scene / [1.,
                               1.,
                               1.,
                               config.MAX_SCALE, config.MAX_SCALE, config.MAX_SCALE,
                               config.MAX_ROTATION,
                               config.MAX_TRANSLATION, config.MAX_TRANSLATION])

    dataset = tf.data.Dataset.zip((image_ds, scenes_ds, angles_ds))

    if only_slice is not None:
        dataset = dataset.take(only_slice)

    # Creating training/validation split
    val_size = int(validation_split * len(dataset))

    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)

    print(f"Creating datasets: training ({len(train_ds)}), validation ({len(val_ds)}).")

    if shuffle:
        val_ds = val_ds.shuffle(1024)
        train_ds = train_ds.shuffle(1024)

    if batch_size is not None:
        val_ds = val_ds.batch(batch_size)
        train_ds = train_ds.batch(batch_size)

    return train_ds, val_ds


if __name__ == "__main__":
    import numpy as np

    tds, vds = get_scene_dataset(config.DS_IMAGES, config.DS_PROGRAMS, batch_size=8, shuffle=False)
    np.set_printoptions(precision=2)
    for img, scs, ang in tds.take(1):
        print(img.shape)
        print(np.max(img[0]))
        print(scs[0].numpy())
        print(ang[0].numpy())

        f, ax = plt.subplots(1, img.shape[-1])
        f.set_size_inches(10, 2)
        ax[0].imshow(img[0, ..., 0])
        ax[1].imshow(img[0, ..., 1])
        ax[2].imshow(img[0, ..., 2])
        if img.shape[-1] > 3:
            ax[3].imshow(img[0, ..., 3])
            ax[4].imshow(img[0, ..., 4])
        plt.show()
