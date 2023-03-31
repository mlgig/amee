import keras
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder


# This wrapper...
# ... adjusts the predict and predict_proba to sklearn standards.
# ... converts univariate input to multivariate.
class KerasWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=None, labels=None):
        self.estimator = estimator
        self.labels = labels

    def predict_proba(self, X):
        return self.estimator.predict(self.prepare_X(X))

    def predict(self, X):
        return self.labels[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def prepare_X(X):
        if np.ndim(X) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            X = np.expand_dims(X, axis=2)
        return X


# This class originated from: https://raw.githubusercontent.com/hfawaz/dl-4-tsc/master/classifiers/resnet.py
class Resnet(KerasWrapper):

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def fit(self, X, y, batch_size=16, epochs=500):
        transformed_X = self.prepare_X(X)
        transformed_y = OneHotEncoder(categories="auto").fit_transform(np.reshape(y, (-1, 1)))

        self.labels = np.unique(y)
        self.build_model(len(self.labels), transformed_X.shape[1:])

        mini_batch_size = int(min(transformed_X.shape[0] / 10, batch_size))
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
        self.estimator.fit(transformed_X, transformed_y, batch_size=mini_batch_size, epochs=epochs,
                           verbose=self.verbose, callbacks=[reduce_lr])

    def build_model(self, n_classes, input_shape):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding="same")(input_layer)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding="same")(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding="same")(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding="same")(input_layer)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation("relu")(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding="same")(output_block_1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding="same")(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding="same")(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding="same")(output_block_1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation("relu")(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding="same")(output_block_2)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding="same")(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding="same")(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation("relu")(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        output_layer = keras.layers.Dense(n_classes, activation="softmax")(gap_layer)

        self.estimator = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.estimator.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
