import tensorflow as tf


class VGGCustom(tf.keras.models.Sequential):
    def __init__(self, input_shape, num_labels):
        super().__init__()

        # 1st Conv Block
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(input_shape)))
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tf.keras.layers.Dropout(0.5))

        # 2nd Conv Block
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(tf.keras.layers.Dropout(0.5))

        # 3rd Conv block
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Fully connected layers
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.add(tf.keras.layers.Dropout(0.2))
        self.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
