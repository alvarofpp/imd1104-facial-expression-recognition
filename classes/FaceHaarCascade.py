import cv2


class FaceHaarCascade:
    def __init__(self):
        self.face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect(self, img):
        return self.face_haar_cascade.detectMultiScale(img, 1.32, 5)
