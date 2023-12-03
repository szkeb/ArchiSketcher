import ctypes
from ctypes import *
from image_processing import *
import numpy as np
import bpy
import config
from network import ArchiSketcher
from blender_script import *
from utils import delete_all_files_from, copy_file

lib = ctypes.CDLL('../SketcherModule/SketcherModule.dll')
N_ITER = 0


def start_starting_canvas():
    lib.StartSession()
    return


def start_next_iteration():
    global N_ITER
    N_ITER += 1
    lib.StartNextSession(N_ITER)
    return


def combine_input(inputs):
    combined = np.concatenate([open_rgba(image)[..., np.newaxis] for image in inputs],
                              axis=-1)
    combined = 1 - combined
    return combined


def predict_scene(net, images, angles, index):
    if images.shape[-1] != config.NUM_OF_OUTPUTS:
        padding = config.NUM_OF_OUTPUTS - images.shape[-1]
        images = np.concatenate([images, np.zeros(shape=(config.IMG_SIZE, config.IMG_SIZE, padding))], axis=-1)
    if angles.shape[0] != config.NUM_OF_OUTPUTS:
        padding = config.NUM_OF_OUTPUTS - angles.shape[-1]
        angles = np.concatenate([angles, np.zeros(shape=padding)], axis=0)

    predictions, raw_outputs = net.predict_scenes((images[np.newaxis, ...], angles[np.newaxis, ...]))

    scene = [{'claster': int(np.rint(shape[0])),
              'bool': int(np.rint(shape[1])),
              'type': int(shape[2]) - 1,
              'scale': shape[3:6],
              'rotation': shape[-3],
              'translation': shape[-2:]} for shape in predictions[index][0] if shape[2] > 0.5]
    real_angles = [angles[0]] if index == 0 else angles[:index+1]
    scene_and_angle = (scene, real_angles)
    return scene_and_angle


def render_predicted_scene(scene, raw_path):
    bpy.ops.wm.open_mainfile(filepath="sketch.blend")
    delete_all_files_from(raw_path)
    render_scenes(scene, config.DISTANCE, raw_path, material='Default')
    copy_file(list(raw_path.glob('*.png'))[-1], Path('app_output') / 'result.png')
    delete_all_files_from(raw_path)
    render_scenes(scene, config.DISTANCE, raw_path, material='Transparent')
    combine_images(raw_path, Path('app_output'), channels=2 + len(scene[0][1]))


def show_result_image(raw_path, inputs):
    shaded = Path("app_output") / 'result.png'
    fig = plt.figure(figsize=(11, 5.5))
    num_of_cells = int(1 + 2*len(inputs))
    axs = np.zeros(num_of_cells, dtype=object)
    gs = fig.add_gridspec(2, 1 + len(inputs))
    axs[0] = fig.add_subplot(gs[:, 0])
    for i in range(len(inputs)):
        axs[1+i] = fig.add_subplot(gs[0, 1+i])
        axs[1+len(inputs)+i] = fig.add_subplot(gs[1, 1+i])
    image = plt.imread(shaded)
    axs[0].imshow(image)
    for i, path in enumerate(inputs):
        original = plt.imread(path)
        axs[1+i].imshow(original)
    for i, path in enumerate(list(raw_path.glob('*.png'))):
        sketch = plt.imread(path)
        axs[1+len(inputs)+i].imshow(sketch)
    plt.show()


if __name__ == '__main__':
    net = ArchiSketcher(config.IMG_SIZE, config.NUM_OF_SHAPES)
    net.load_weights(config.MODEL_SAVE_PATH)

    angles = np.asarray([np.pi/4])
    raw_path = Path('app_output') / 'raw'
    # Creating the input images
    start_starting_canvas()
    # Merging the input images
    input_paths = ["Top.png", "Side.png", "Persp_0.png"]
    input_images = combine_input(input_paths)

    print("Running neural network model...")

    # Running deep learning model on the input
    result = predict_scene(net, input_images, angles, N_ITER)
    # Render prediction
    render_predicted_scene([result], raw_path)
    # Show result
    show_result_image(raw_path, input_paths)

    for i in range(config.NUM_OF_ANGLES - 1):
        want_more = input("Do you want to enter a new perspective view? [y/n]")
        if want_more != "y":
            exit()
        new_angle = float(input("What's the angle of the new view? "))
        angles = np.concatenate([angles, [new_angle * np.pi / 180]], axis=0)

        # Starting next iteration
        start_next_iteration()
        # Merging new inputs
        input_paths.append(f"Persp_{1+i}.png")
        input_images = combine_input(input_paths)

        print("Running neural network model...")

        # Running deep learning model on the input
        result = predict_scene(net, input_images, angles, N_ITER)
        # Render prediction
        render_predicted_scene([result], raw_path)
        # Show result
        show_result_image(raw_path, input_paths)

