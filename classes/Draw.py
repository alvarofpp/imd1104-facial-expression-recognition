import cv2
from PIL import ImageDraw


class Draw:
    # define RGB color constants
    COLOR_ANGRY = (234, 59, 70)
    COLOR_DISGUST = (216, 91, 99)
    COLOR_FEAR = (92, 92, 92)
    COLOR_HAPPY = (192, 186, 128)
    COLOR_SAD = (100, 193, 232)
    COLOR_SURPRISE = (214, 128, 173)
    COLOR_NEUTRAL = (253, 196, 125)
    WHITE = (255, 255, 255)

    # map each emotion to the color used for the bounding box
    EMOTION_COLOR_MAP = {
        'angry': COLOR_ANGRY,
        'disgust': COLOR_DISGUST,
        'fear': COLOR_FEAR,
        'happy': COLOR_HAPPY,
        'sad': COLOR_SAD,
        'surprise': COLOR_SURPRISE,
        'neutral': COLOR_NEUTRAL,
    }

    @staticmethod
    def rectangle(captured_image, x, y, w, h):
        cv2.rectangle(captured_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

    @staticmethod
    def text(captured_image, predict, coordinates):
        cv2.putText(captured_image, predict[0], coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    @staticmethod
    def draw(image_pixels, predict_emotion, box, font_loader=None, padding=0):
        #draw = ImageDraw.Draw(image_pixels)
        draw = ImageDraw.Draw(image_pixels)

        message = predict_emotion
        color = Draw.EMOTION_COLOR_MAP[predict_emotion]

        # get the face extents from the bounding box
        x_min, y_min, x_max, y_max = box

        # draw the main bounding box outline for the face
        draw.rectangle([x_min, y_min, x_min+x_max, y_min+y_max], outline=color)

        if font_loader is not None:
            font = font_loader(x_max - x_min)

        # get the extents of the text message
        ascent, descent = font.getmetrics()
        text_width = font.getsize(message)[0]
        text_height = ascent + descent

        # draw the message over a filled-in background rectangle
        draw.rectangle((x_min, y_min, x_min + text_width + padding, y_min + text_height + padding), fill=color)
        draw.text((x_min + padding, y_min + padding), message, font=font, fill=Draw.WHITE)
