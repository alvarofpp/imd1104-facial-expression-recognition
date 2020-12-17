import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json


class Model:
    def __init__(self):
        self._model = None
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        self.custom_objects = {
            'DeepCNN': tf.keras.models.Sequential,
            'DeepCNNCustom': tf.keras.models.Sequential,
            'ShallowCNN': tf.keras.models.Sequential,
            'VGGA11': tf.keras.models.Sequential,
            'VGGALRN11': tf.keras.models.Sequential,
            'VGGB13': tf.keras.models.Sequential,
            'VGGC16': tf.keras.models.Sequential,
            'VGGD16': tf.keras.models.Sequential,
            'VGGE19': tf.keras.models.Sequential,
            'VGGE19Custom': tf.keras.models.Sequential,
        }

    def predict(self, img) -> str:
        predictions = self._model.predict(img)
        max_index = np.argmax(predictions[0])
        return self.emotions[max_index]
    
    def load(self, name):
        json_file = open('models/{}.json'.format(name), 'r').read()
        self._model = model_from_json(json_file, custom_objects=self.custom_objects)
        self._model.load_weights('models/{}.h5'.format(name))

    @staticmethod
    def preprocessing(gray_image, x_min, y_min, x_max, y_max):
        #roi_gray = gray_image[y:y + w, x:x + h]
        roi_gray = gray_image[y_min:y_min + x_max, x_min:x_min + y_max]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        return img_pixels
