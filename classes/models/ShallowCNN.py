import tensorflow as tf


class ShallowCNN(tf.keras.models.Sequential):
    def __init__(self, input_shape, num_labels):
        super().__init__()

        # 1st Convolutional Layer
        self.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(input_shape)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Activation('relu'))

        # 2nd Convolutional Layer
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(tf.keras.layers.Activation('relu'))

        # Flattening
        self.add(tf.keras.layers.Flatten())

        # Fully connected layers
        self.add(tf.keras.layers.Dense(512))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
