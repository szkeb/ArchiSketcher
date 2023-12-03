import config
from dataset import get_scene_dataset
from network import ArchiSketcher
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import backend as K


if __name__ == '__main__':
    ##################
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 100
    LOAD_MODEL = False
    ##################

    train_ds, val_ds = get_scene_dataset(config.DS_IMAGES,
                                         config.DS_PROGRAMS,
                                         batch_size=BATCH_SIZE,
                                         validation_split=0.1,
                                         only_slice=16384)

    net = ArchiSketcher(config.IMG_SIZE, config.NUM_OF_SHAPES)
    if LOAD_MODEL:
        net.load_weights(config.MODEL_SAVE_PATH)
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR))
    K.set_value(net.optimizer.learning_rate, LR)

    net.fit(
        train_ds,
        shuffle=True,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.MODEL_SAVE_PATH, save_weights_only=True, monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=1e-4, patience=2, factor=0.5)])
