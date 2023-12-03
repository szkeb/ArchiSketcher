from pathlib import Path
from numpy import pi as PI

IMG_SIZE = 256

MODEL_SAVE_PATH = Path('saved_models/multipleAngle/dla')

# Dataset constants
DS_SIZE = 8192
BLENDER_DIR = Path('datasets/generatedBlenderScenes')
DS_DIR = Path('datasets/multipleAngle')
DS_IMAGES = DS_DIR / 'combinedScenes'
DS_PROGRAMS = DS_DIR / 'scenes.txt'
DS_PROGRAMS_BLENDER = BLENDER_DIR / 'scenes.txt'
DS_DIR_TEST = Path('datasets/sceneDataset_test')
DS_IMAGES_TEST = DS_DIR_TEST / 'combinedScenes'
DS_PROGRAMS_TEST = DS_DIR_TEST / 'scenes.txt'

# Eval path
EVAL_DIR = Path('eval')
EVAL_DIR_ITERATIVE = EVAL_DIR / 'iterative'
EVAL_TMP_DIR = EVAL_DIR / 'raw_predictions'
EVAL_TMP_DIR_ITERATIVE = EVAL_DIR_ITERATIVE / 'raw_predictions'
EVAL_SCENES_PATH = EVAL_DIR / 'eval_scenes_predicted.txt'
EVAL_SCENES_PATH_ITERATIVE = EVAL_DIR_ITERATIVE / 'eval_scenes_predicted.txt'
EVAL_ANGLES_PATH = EVAL_DIR / 'eval_angles_original.txt'
EVAL_ANGLES_PATH_ITERATIVE = EVAL_DIR_ITERATIVE / 'eval_angles_original.txt'
EVAL_BATCH = 16

# Generating
NUM_OF_SHAPES = 5
NUM_OF_PRIMITIVE_TYPES = 3  # +1 for marking non visible
NUM_OF_PARAMETERS = 3 + 1 + 2  # Number of parameters of a single object: sx, sy, sz, r, tx, ty
NUM_OF_ANGLES = 3
ANGLE_STEPS = 32
MIN_ANGLE_OFFSET = 3
DISTANCE = 30
MAX_SCALE = 5.5
MAX_TRANSLATION = 25.
MAX_ROTATION = PI / 4.
SCALE_STEP = 0.5
ROTATION_STEP = PI / 8.
TRANSLATION_STEP = 1.

NUM_OF_OUTPUTS = NUM_OF_ANGLES + 2