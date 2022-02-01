import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

prev_time = 0
cur_time = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=False)
    land_mark_list = detector.find_position(img, draw=False)
    if len(land_mark_list) != 0:
        print(land_mark_list[4])

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('Dolboeb na ekrane', img)
    cv2.waitKey(1)
