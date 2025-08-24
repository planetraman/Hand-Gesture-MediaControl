import cv2
import mediapipe as mp
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# Webcam setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)   # change to 0 if you use laptop cam
cap.set(3, wCam)
cap.set(4, hCam)

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()  # usually (-65.25, 0.0, 0.03125)
print("Volume range:", vol_range)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    left_index, right_index = None, None

    if results.multi_hand_landmarks and results.multi_handedness:
        # Pair each hand with its label (Left/Right)
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # "Left" or "Right"
            h, w, c = img.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if label == "Left":
                left_index = (x, y)
            else:
                right_index = (x, y)

        # --- Brightness Control (Thumb tip = 4, Middle tip = 12) ---

        if label == "Left":
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            middle_x, middle_y = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)

            # Draw thumb & middle
            cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (middle_x, middle_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (middle_x, middle_y), (0, 255, 0), 2)

            # Distance → brightness
            bright_len = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
            if bright_len > 20:   # ignore small noise
                brightness = int(np.interp(bright_len, [20, 250], [0, 100]))
                sbc.set_brightness(brightness)

                cv2.putText(img, f'Brightness: {brightness} %', (40, 150),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)


    # Proceed only if both hands are detected
    if left_index and right_index:
        x1, y1 = left_index
        x2, y2 = right_index

        # Draw points + line
        cv2.circle(img, (x1, y1), 12, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Midpoint
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 12, (255, 255, 255), cv2.FILLED)

        # Distance
        length = math.hypot(x2 - x1, y2 - y1)

        if length > 30:  # ignore false duplicate hands
            vol_scalar = np.interp(length, [50, 400], [0.0, 1.0])
            vol_scalar = np.clip(vol_scalar, 0.0, 1.0)
            volume.SetMasterVolumeLevelScalar(vol_scalar, None)

            # Show volume %
            vol_percent = int(vol_scalar * 100)
            cv2.putText(img, f'Vol: {vol_percent} %', (40, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
