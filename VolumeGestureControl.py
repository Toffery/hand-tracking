import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam_width, cam_height = 1280, 720
prev_time = 0

detector = htm.HandDetector(detect_confidence=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()

min_volume = volume_range[0]
max_volume = volume_range[1]

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)


while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame. If this message continues to appear, please restart the program")
        continue

    img = detector.find_hands(img, draw=False)
    land_mark_list = detector.find_position(img, draw=False)
    # print(land_mark_list)
    if len(land_mark_list) != 0:
        # print(land_mark_list[4], land_mark_list[8])
        x1, y1 = land_mark_list[4][1], land_mark_list[4][2]
        x2, y2 = land_mark_list[8][1], land_mark_list[8][2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 10, (0, 0, 255), cv2.FILLED)

        length = hypot(x2 - x1, y2 - y1)
        print(length)
        # maximum = 200
        # minimum = 50

        vol = np.interp(length, [30, 410], [-20, max_volume])
        print(vol)
        if length < 150:
            volume.SetMasterVolumeLevel(-30, None)
        if length < 100:
            volume.SetMasterVolumeLevel(-35, None)
        if length < 50:
            volume.SetMasterVolumeLevel(-50, None)
        if length < 30:
            volume.SetMasterVolumeLevel(-63.5, None)
        if length > 150:
            volume.SetMasterVolumeLevel(vol, None)

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Window', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
