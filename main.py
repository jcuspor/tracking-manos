import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
import pyautogui

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

def draw_landmarks(img, detection_results):
    img_copy = np.copy(img)

    for hand_landmarks in detection_results.hand_landmarks:
        mp_drawing.draw_landmarks(
            img_copy,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        index_tip = hand_landmarks[8].y
        middle_tip = hand_landmarks[12].y
        thumb_tip = hand_landmarks[4].y

        if index_tip > thumb_tip:
            pyautogui.press("volumeup", 1, 0.2)
        elif middle_tip > thumb_tip:
            pyautogui.press("volumedown", 1, 0.2)

    return img_copy

def detect_hands(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = detector.detect(img)
    drawn_image = draw_landmarks(img.numpy_view(), detection_result)

    return cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = detect_hands(frame)
        img = cv2.flip(img, 1)
        cv2.imshow("img", img)

        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()