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

render = True

if __name__ == '__main__':
    ds, _ = get_scene_dataset(config.DS_IMAGES,
                              config.DS_PROGRAMS,
                              batch_size=config.EVAL_BATCH,
                              validation_split=0.0,
                              only_slice=1024)

    net = ArchiSketcher(config.IMG_SIZE, config.NUM_OF_SHAPES)
    net.load_weights(config.MODEL_SAVE_PATH)

    np.set_printoptions(precision=2)
    scenes = []
    normalized_angles = []
    for data in ds.take(1):
        img, true, angles = data

        scene_iterations, raw_outputs = net.predict_scenes((img, angles))
        print(len(raw_outputs))
        print(angles)
        print(raw_outputs[0].shape)
        print(raw_outputs[0][:, :, -6:])
        print(raw_outputs)
        # only the last iteration
        scenes = scene_iterations[-1]

        img3 = img[..., :3].numpy()
        for i, pic in enumerate(img3):
            filename = 'original_' + pad_string(str(i), img3) + '.png'
            plt.imsave(config.EVAL_DIR / filename, pic)

        # Normalize back
        angles *= (np.pi * 2.)
        normalized_angles = angles.numpy()
        break

    bpy.ops.wm.open_mainfile(filepath="sketch.blend")

    scenes = [
        [{'claster': int(np.rint(shape[0])),
          'bool': int(np.rint(shape[1])),
          'type': int(shape[2]) - 1,
          'scale': shape[3:6],
          'rotation': shape[-3],
          'translation': shape[-2:]} for shape in scene if shape[2] > 0.5 and shape[0] > 0.5]
        for scene in scenes]

    angles = np.reshape(normalized_angles, (config.EVAL_BATCH, config.NUM_OF_ANGLES))
    angles = np.repeat(angles[np.newaxis, ...], config.NUM_OF_OUTPUTS, axis=0)
    angles = np.reshape(angles, (config.EVAL_BATCH * config.NUM_OF_OUTPUTS, config.NUM_OF_ANGLES))
    scenes_and_angles = zip(scenes, angles)

    delete_all_files_from(config.EVAL_TMP_DIR)
    blender_script.render_scenes(scenes_and_angles, config.DISTANCE, config.EVAL_TMP_DIR, material='Transparent')
    combine_images(config.EVAL_TMP_DIR, config.EVAL_DIR, prefix='predicted', channels=3, starting_image_idx=0)

    N = 10
    S = 2
    num_of_outputs = 3
    f, axs = plt.subplots(2, N)
    f.set_size_inches(S * N, S * 2)

    filenames = sorted(os.listdir(config.EVAL_DIR))
    originals = [f for f in filenames if '.png' in f and 'original' in f]
    predicted = [f for f in filenames if '.png' in f and 'predicted' in f]

    idx_list = list(range(config.EVAL_BATCH)[:N])

    for i_row, idx in enumerate(idx_list):
        o_file = originals[idx]
        o_img = plt.imread(config.EVAL_DIR / o_file)

        p_file = predicted[idx]
        p_img = plt.imread(config.EVAL_DIR / p_file)

        axs[0, i_row].imshow(o_img)
        axs[1, i_row].imshow(p_img)

    plt.show()
