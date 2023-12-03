import numpy as np
from matplotlib import pyplot as plt

import config
from dataset import get_scene_dataset
from network import ArchiSketcher
from utils import pad_string, delete_all_files_from
import blender_script
import os
import bpy
from image_processing import combine_images
import tensorflow as tf

render = True

if __name__ == '__main__':
    ds, _ = get_scene_dataset(config.DS_IMAGES,
                              config.DS_PROGRAMS,
                              batch_size=config.EVAL_BATCH,
                              validation_split=0.0,
                              only_slice=1024)

    net = ArchiSketcher(config.IMG_SIZE, config.NUM_OF_SHAPES)
    net.load_weights(config.MODEL_SAVE_PATH)

    while True:
        np.set_printoptions(precision=2)
        scene_iterations = []
        normalized_angles = []
        losses = []
        for data in ds.take(1):
            img, true, angles = data

            batch_iterations, raw_outputs = net.predict_scenes((img, angles), True)
            # Only one prediction from the batch, for every iteration
            scene_iterations = [batch[0] for batch in batch_iterations]
            for output in raw_outputs:
                losses.append(net.custom_loss(true[0][tf.newaxis, ...], output[0][tf.newaxis, ...]).numpy())

            img3 = img[0, ..., :3].numpy()
            filename = 'original.png'
            plt.imsave(config.EVAL_DIR_ITERATIVE / filename, img3)

            # Normalize back
            angles *= (np.pi * 2.)
            normalized_angles = angles[0].numpy()

            break

        bpy.ops.wm.open_mainfile(filepath="sketch.blend")

        scenes = [
            [{'claster': int(np.rint(shape[0])),
              'bool': int(np.rint(shape[1])),
              'type': int(shape[2]) - 1,
              'scale': shape[3:6],
              'rotation': shape[-3],
              'translation': shape[-2:]} for shape in scene if shape[2] > 0.5 and shape[0] > 0.5]
            for scene in scene_iterations]

        angles = np.repeat(normalized_angles[np.newaxis, ...], config.NUM_OF_OUTPUTS, axis=0)
        angles = np.reshape(angles, (config.NUM_OF_OUTPUTS, config.NUM_OF_ANGLES))
        scenes_and_angles = zip(scenes, angles)

        delete_all_files_from(config.EVAL_TMP_DIR_ITERATIVE)
        blender_script.render_scenes(scenes_and_angles, config.DISTANCE, config.EVAL_TMP_DIR_ITERATIVE, material='Transparent')
        combine_images(config.EVAL_TMP_DIR_ITERATIVE, config.EVAL_DIR_ITERATIVE, prefix='predicted', channels=3, starting_image_idx=0)

        N = config.NUM_OF_ANGLES
        S = 2
        f, axs = plt.subplots(2, N)
        f.set_size_inches(S * N, S * 2)

        filenames = sorted(os.listdir(config.EVAL_DIR_ITERATIVE))
        original = [f for f in filenames if '.png' in f and 'original' in f][0]
        predicteds = [f for f in filenames if '.png' in f and 'predicted' in f]

        for i in range(N):
            o_img = plt.imread(config.EVAL_DIR_ITERATIVE / original)

            p_file = predicteds[i]
            p_img = plt.imread(config.EVAL_DIR_ITERATIVE / p_file)

            axs[0, i].imshow(o_img)
            axs[1, i].imshow(p_img)

            print(f'Loss {i}: {losses[i]}')

        plt.show()
