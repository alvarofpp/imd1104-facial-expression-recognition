import cv2
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from classes import Draw, FaceHaarCascade, FontLoader, Model


def print_table(emotions_count):
    emotions = []
    counts = []

    for emotion, count in emotions_count.items():
        if count > 0:
            emotions.append(emotion)
            counts.append(count)

    df_emotions_count = pd.DataFrame(list(zip(emotions, counts)), columns=['Emotion', 'Count'])
    df_emotions_count = df_emotions_count.sort_values('Count', ascending=False)
    df_emotions_count = df_emotions_count.reset_index(drop=True, col_level=1)
    df_emotions_count.index = np.arange(1, len(df_emotions_count) + 1)
    st.table(df_emotions_count)


def main():
    st.title('Facial Expression Recognition')

    st.markdown("""
    By: [Álvaro Ferreira Pires de Paiva](https://github.com/alvarofpp).
    - Repository: [https://github.com/alvarofpp/imd1104-facial-expression-recognition](https://github.com/alvarofpp/imd1104-facial-expression-recognition)
    - Medium: (pt-br) [Reconhecimento de expressão facial usando CNN](#)

    You can use the sidebar to upload the image and choose the CNN model you want to apply.
    """)
    st.markdown('----------')

    model_files = {
        'Deep': 'DeepCNN',
        'Deep (Custom)': 'DeepCNNCustom',
        'Shallow': 'ShallowCNN',
        'VGG-11 (A)': 'VGGA11',
        'VGG-11 (A-LRN)': 'VGGALRN11',
        'VGG-13 (B)': 'VGGB13',
        'VGG-16 (C)': 'VGGC16',
        'VGG-16 (D)': 'VGGD16',
        'VGG-19 (E)': 'VGGE19',
        'VGG-19 (Custom)': 'VGGE19Custom',
    }

    image_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    choice = st.sidebar.selectbox('Model', list(model_files.keys()))

    face_haar_cascade = FaceHaarCascade()
    model = Model()
    model.load(model_files[choice])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.markdown('## Original image')
        st.image(our_image, width=700)

        st.markdown('## Faces with emotions')
        captured_image = np.array(our_image)
        gray_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detect(gray_image)

        st.text('{} faces were found in the image:'.format(len(faces_detected)))
        emotions_count = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0,
        }

        for (x_min, y_min, x_max, y_max) in faces_detected:
            image_pixels = Model.preprocessing(gray_image, x_min, y_min, x_max, y_max)
            predict_emotion = model.predict(image_pixels)
            emotions_count[predict_emotion] += 1
            Draw.draw(our_image, predict_emotion, (x_min, y_min, x_max, y_max), font_loader=FontLoader())

        print_table(emotions_count)
        st.image(our_image, width=700)


if __name__ == '__main__':
    main()
