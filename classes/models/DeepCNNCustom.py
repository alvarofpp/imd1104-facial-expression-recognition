import tensorflow as tf


class DeepCNNCustom(tf.keras.models.Sequential):
    def __init__(self, input_shape, num_labels):
        super().__init__()

        # 1st Conv Block
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), input_shape=(input_shape)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(tf.keras.layers.Activation('relu'))

        # 2nd Conv Block
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(tf.keras.layers.Activation('relu'))

        # 2nd Conv Block
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(tf.keras.layers.Activation('relu'))

        # 3rd Conv block
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.3))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.add(tf.keras.layers.Activation('relu'))

        # 4th Conv block
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.3))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.add(tf.keras.layers.Activation('relu'))

        # Flattening
        self.add(tf.keras.layers.Flatten())

        # Fully connected layers
        self.add(tf.keras.layers.Dense(256))
        #self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.Activation('relu'))

        # Fully connected layer 2nd layer
        self.add(tf.keras.layers.Dense(512))
        #self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.Activation('relu'))

        self.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
