import tensorflow as tf
import config
layers = tf.keras.layers
import numpy as np


class ArchiSketcher(tf.keras.Model):
    def __init__(
            self,
            image_size,
            num_of_shapes,
            only_first=False,
            name="ArchiSketcher",
            **kwargs):
        super(ArchiSketcher, self).__init__(name=name, **kwargs)

        self.img_size = image_size
        self.input_dim = (image_size, image_size, 3)
        self.num_of_shapes = num_of_shapes
        self.num_of_angles = config.NUM_OF_ANGLES

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.claster_tracker = tf.keras.metrics.Mean(name="cidx")
        self.bool_tracker = tf.keras.metrics.Mean(name="bool")
        self.bool_acc_tracker = tf.keras.metrics.BinaryAccuracy(name="bacc")
        self.shape_tracker = tf.keras.metrics.Mean(name="shp")
        self.claster_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="cacc")
        self.shape_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="sacc")
        self.translation_tracker = tf.keras.metrics.Mean(name="pos")
        self.rotation_tracker = tf.keras.metrics.Mean(name="rot")
        self.scale_tracker = tf.keras.metrics.Mean(name="scl")
        self.sorting_tracker = tf.keras.metrics.Mean(name="srt")

        # --- LAYERS ---

        # Encoder component
        # Image -> feature vector
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=self.input_dim),
            tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3),
                pooling='avg')
        ], name='encoder')

        self.intermediate_dim = 256

        self.angle_integrator_top = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.intermediate_dim, activation='relu'),
        ], name='angle_integrator_top')
        self.angle_integrator_side = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.intermediate_dim, activation='relu'),
        ], name='angle_integrator_side')
        self.angle_integrator_persp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.intermediate_dim, activation='relu'),
        ], name='angle_integrator_persp')

        self.initial_view_merger = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.intermediate_dim, activation='relu')
        ], name='initial_merger')

        self.feature_output_merger = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.intermediate_dim, activation='relu'),
        ], name='feature_output_merger')

        self.output_matrix_shape = (config.NUM_OF_SHAPES, config.NUM_OF_PRIMITIVE_TYPES + 1 + config.NUM_OF_SHAPES + 1 + 1 + config.NUM_OF_PARAMETERS)

        regularization = tf.keras.regularizers.l2(1e-2)
        param_matrix_size = config.NUM_OF_SHAPES * config.NUM_OF_PARAMETERS
        shape_matrix_size = config.NUM_OF_SHAPES * (config.NUM_OF_PRIMITIVE_TYPES + 1)
        claster_matrix_size = config.NUM_OF_SHAPES * (config.NUM_OF_SHAPES + 1)
        bool_matrix_size = config.NUM_OF_SHAPES * 1

        self.parameter_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(param_matrix_size, activation='sigmoid', kernel_regularizer=regularization),
            tf.keras.layers.Reshape((config.NUM_OF_SHAPES, config.NUM_OF_PARAMETERS))
        ], name='parameter_head')

        self.shape_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=regularization),
            tf.keras.layers.Dense(units=shape_matrix_size, activation='relu', kernel_regularizer=regularization),
            tf.keras.layers.Reshape((config.NUM_OF_SHAPES, config.NUM_OF_PRIMITIVE_TYPES + 1)),
            tf.keras.layers.Softmax(axis=-1)
        ], name='shape_head')

        self.claster_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=regularization),
            tf.keras.layers.Dense(units=claster_matrix_size, activation='relu', kernel_regularizer=regularization),
            # Num of shapes + 1 for zero claster
            tf.keras.layers.Reshape((config.NUM_OF_SHAPES, config.NUM_OF_SHAPES + 1)),
            tf.keras.layers.Softmax(axis=-1)
        ], name='claster_head')

        self.bool_head = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_regularizer=regularization),
            tf.keras.layers.Dense(units=bool_matrix_size, activation='sigmoid', kernel_regularizer=regularization),
            # Num of shapes + 1 for zero claster
            tf.keras.layers.Reshape((config.NUM_OF_SHAPES, 1))
        ], name='bool_head')

        self.output_flatter = tf.keras.layers.Flatten()

        self.only_first_part = only_first

    @tf.function
    def call(self, data, training=False):
        img, angles = data

        # Decomposing input to top/side and perspective image
        top = img[..., 0][..., tf.newaxis]
        side = img[..., 1][..., tf.newaxis]
        perspectives = [img[..., 2 + i][..., tf.newaxis] for i in range(self.num_of_angles)]

        top = tf.image.grayscale_to_rgb(top)
        side = tf.image.grayscale_to_rgb(side)
        perspectives = [tf.image.grayscale_to_rgb(p) for p in perspectives]

        # Encoding feature vectors
        features_top = self.encoder(top, training)
        features_side = self.encoder(side, training)
        inter_top = self.angle_integrator_top(features_top, training)
        inter_side = self.angle_integrator_side(features_side, training)

        inters_persp = []
        for i in range(self.num_of_angles):
            p = perspectives[i]
            features = self.encoder(p, training)
            features_and_angle = tf.concat([features, angles[..., i][..., tf.newaxis]], axis=-1)
            inter_persp = self.angle_integrator_persp(features_and_angle, training)
            inters_persp.append(inter_persp)

        initial_views = tf.concat([inter_top, inter_side, inters_persp[0]], axis=-1)
        initial_intermediate = self.initial_view_merger(initial_views)
        intermediates = [initial_intermediate, *inters_persp[1:]]

        outputs = []
        outputs.append(tf.zeros(shape=(tf.shape(img)[0], *self.output_matrix_shape), dtype=tf.float32))
        for i, inter_persp in enumerate(intermediates):
            #enriched_inter_input = tf.concat([inter_top, inter_side, inter_persp], axis=-1)
            #enriched_inter = self.initial_view_merger(enriched_inter_input, training)
            flat_prev_output = tf.keras.layers.Flatten()(outputs[i])
            features_and_output = tf.concat([flat_prev_output, tf.keras.layers.Flatten()(inter_persp)], axis=-1)
            intermediate = self.feature_output_merger(features_and_output, training)

            output_params = self.parameter_head(intermediate, training)
            output_shapes = self.shape_head(intermediate, training)
            first_phase = tf.concat([output_shapes, output_params], axis=-1)

            if self.only_first_part:
                outputs.append(first_phase)
                continue

            features_and_first_phase = tf.concat([intermediate, self.output_flatter(first_phase)], axis=-1)
            output_claster = self.claster_head(features_and_first_phase, training)
            output_bool = self.bool_head(features_and_first_phase, training)

            output = tf.concat([output_claster, output_bool, output_shapes, output_params], axis=-1)
            outputs.append(output)

        return outputs[1:]

    @tf.function
    def train_step(self, data):
        img, y_true, angle = data

        with tf.GradientTape() as tape:
            y_pred = self((img, angle), training=True)  # Forward pass
            # Compute our own loss
            loss = self.custom_loss_multiple(y_true, y_pred)

        # Compute gradients
        trainable_vars = self.encoder.trainable_variables \
                           + self.angle_integrator_top.trainable_variables \
                           + self.angle_integrator_side.trainable_variables \
                           + self.angle_integrator_persp.trainable_variables \
                           + self.initial_view_merger.trainable_variables \
                           + self.feature_output_merger.trainable_variables \
                           + self.parameter_head.trainable_variables \
                           + self.shape_head.trainable_variables
        if not self.only_first_part:
            trainable_vars += self.claster_head.trainable_variables \
            + self.bool_head.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        img, y_true, angle = data

        y_pred = self((img, angle), training=False)
        loss = self.custom_loss_multiple(y_true, y_pred)

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def custom_loss_multiple(self, y_true, y_preds):
        loss = 0
        factor = 1.0
        silent = False
        for i in reversed(range(len(y_preds))):
            y_pred = y_preds[i]
            loss += (factor * self.custom_loss(y_true, y_pred, silent=silent))
            factor *= 0.4
            silent = True

        return loss

    @tf.function
    def custom_loss(self, y_true, y_pred, silent=False):
        num_of_clasters = config.NUM_OF_SHAPES + 1
        claster_true = y_true[:, :, 0]
        claster_pred = y_pred[:, :, :num_of_clasters]

        bool_idx = num_of_clasters
        bool_true = y_true[:, :, 1]
        bool_pred = y_pred[:, :, bool_idx]

        shape_start = bool_idx + 1
        shape_end = shape_start + config.NUM_OF_PRIMITIVE_TYPES + 1
        shape_true = y_true[:, :, 2]
        shape_pred = y_pred[:, :, shape_start:shape_end]

        scale_true = y_true[:, :, -6:-3]
        scale_pred = y_pred[:, :, -6:-3]
        rotation_true = y_true[:, :, -3]
        rotation_pred = y_pred[:, :, -3]
        translation_true = y_true[:, :, -2:]
        translation_pred = y_pred[:, :, -2:]

        shape_loss = self.shape_loss(shape_true, shape_pred, silent)
        scale_loss = self.scale_loss(scale_true, scale_pred, silent)
        rotation_loss = self.rotation_loss(rotation_true, rotation_pred, silent)
        translation_loss = self.translation_loss(translation_true, translation_pred, silent)
        sorting_loss = self.sorting_loss(translation_pred, silent)

        loss = shape_loss + translation_loss + rotation_loss + scale_loss + sorting_loss
        if self.only_first_part:
            return loss
        else:
            claster_loss = self.claster_loss(claster_true, claster_pred, silent)
            bool_loss = self.bool_loss(bool_true, bool_pred, silent)
            return loss + 0.6 * (claster_loss + bool_loss)

    @tf.function
    def claster_loss(self, true, pred, silent):
        error = tf.keras.losses.sparse_categorical_crossentropy(true, pred)
        error = tf.reduce_mean(error, axis=-1)

        if not silent:
            self.claster_tracker.update_state(error)
            self.claster_acc_tracker.update_state(true, pred)

        return error

    @tf.function
    def bool_loss(self, true, pred, silent):
        error = tf.keras.losses.binary_crossentropy(true, pred)

        if not silent:
            self.bool_tracker.update_state(error)
            self.bool_acc_tracker.update_state(true, pred)

        return error

    @tf.function
    def shape_loss(self, true, pred, silent):
        error = tf.keras.losses.sparse_categorical_crossentropy(true, pred)
        error = tf.reduce_mean(error, axis=-1)

        if not silent:
            self.shape_tracker.update_state(error)
            self.shape_acc_tracker.update_state(true, pred)

        return error

    @tf.function
    def translation_loss(self, true, pred, silent):
        error = tf.reduce_sum(tf.square(true - pred), axis=[-2, -1])

        if not silent:
            self.translation_tracker.update_state(error)

        return error

    @tf.function
    def sorting_loss(self, pred, silent):
        pred_distance = tf.square(pred[..., 0]) + tf.square(pred[..., 1])
        pred_args = tf.argsort(pred_distance, direction='DESCENDING')
        error = tf.reduce_sum(tf.square(tf.range(config.NUM_OF_SHAPES) - pred_args), axis=-1)

        if not silent:
            self.sorting_tracker.update_state(error)

        return tf.cast(error, dtype=tf.float32)

    @tf.function
    def rotation_loss(self, true, pred, silent):
        error = tf.reduce_sum(tf.square(true - pred), axis=[-1])

        if not silent:
            self.rotation_tracker.update_state(error)

        return error

    @tf.function
    def scale_loss(self, true, pred, silent):
        error = tf.reduce_sum(tf.square(true - pred), axis=[-2, -1])

        if not silent:
            self.scale_tracker.update_state(error)

        return error

    def get_output_descriptor(self, output_slice):
        y = output_slice
        num_of_clasters = config.NUM_OF_SHAPES + 1
        y_claster = y[:, :, :num_of_clasters]
        y_claster = np.argmax(y_claster, axis=-1)
        y_claster = y_claster.astype(np.float32)

        bool_idx = num_of_clasters
        y_bool = y[:, :, bool_idx]

        shape_start = bool_idx + 1
        shape_end = shape_start + config.NUM_OF_PRIMITIVE_TYPES + 1
        y_shapes = y[:, :, shape_start:shape_end]
        y_shapes = np.argmax(y_shapes, axis=-1)
        y_shapes = y_shapes.astype(np.float32)

        params_start = -(3 + 1 + 2)
        y_params = y[:, :, params_start:]

        y = np.concatenate([y_claster[..., np.newaxis], y_bool[..., np.newaxis], y_shapes[..., np.newaxis], y_params], axis=-1)

        y = y * [1,
                 1,
                 1,
                 config.MAX_SCALE, config.MAX_SCALE, config.MAX_SCALE,
                 config.MAX_ROTATION,
                 config.MAX_TRANSLATION, config.MAX_TRANSLATION]

        return y

    def predict_scenes(self, input, training=True):
        raw_outputs = self(input, training=training)
        # Scene descriptors, and raw outputs
        return [self.get_output_descriptor(y) for y in raw_outputs], raw_outputs

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker,
            self.claster_acc_tracker,
            self.bool_acc_tracker,
            self.shape_acc_tracker,
            self.translation_tracker,
            self.rotation_tracker,
            self.scale_tracker,
            self.sorting_tracker]