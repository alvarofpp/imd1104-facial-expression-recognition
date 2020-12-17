import tensorflow as tf


class VGGE19Custom(tf.keras.models.Sequential):
    def __init__(self, input_shape, num_labels):
        super().__init__()

        # 1st Conv Block
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(input_shape)))
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 2nd Conv Block
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 3rd Conv block
        self.add(tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 4th Conv block
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(5, 5), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 5th Conv block
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(5, 5), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Fully connected layers
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
