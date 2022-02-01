import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import numpy as np
import os

folder_path = 'Header'
my_list = sorted(os.listdir(folder_path))
print(my_list)
overlay_list = []
for im_path in my_list:
    image = cv2.imread(f'{folder_path}/{im_path}')
    overlay_list.append(image)

img_canvas = np.zeros((720, 1280, 3), np.uint8)
header = overlay_list[0]
draw_color = (0, 0, 255)
brush_thickness = 15
x_prev = 0
y_prev = 0
cam_width, cam_height = 1280, 720
prev_time = 0

detector = htm.HandDetector()

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame. If this message continues to appear, please restart the program")
        continue
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=False)
    land_mark_list = detector.find_position(img, draw=False)

    if len(land_mark_list) != 0:

        fingers = []
        x1, y1 = land_mark_list[8][1:]
        x2, y2 = land_mark_list[12][1:]
        print(x1, x2)
        print(y1, y2)

        fingers = detector.fingers_up()
        # print(fingers)

        # Selection Mode
        if fingers[1] and fingers[2]:
            x_prev = 0
            y_prev = 0
            # cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (0, 0, 255), cv2.FILLED)
            print('Selection mode')
            if y1 < 126:
                if 0 < x1 < 160:  # Click first colour (orange #FFAA00 (0, 170, 255))
                    header = overlay_list[1]
                    draw_color = (0, 170, 255)
                    brush_thickness = 15
                if 160 < x1 < 320:  # Click second colour (pink #E6399B (155, 57, 230))
                    header = overlay_list[2]
                    draw_color = (155, 57, 230)
                    brush_thickness = 15
                if 320 < x1 < 480:  # Click third colour (red #FF0000 (0, 0, 255))
                    header = overlay_list[3]
                    draw_color = (0, 0, 255)
                    brush_thickness = 15
                if 480 < x1 < 640:  # Click fourth colour (yellow #FFFF00 (0, 255, 255))
                    header = overlay_list[4]
                    draw_color = (0, 255, 255)
                    brush_thickness = 15
                if 640 < x1 < 800:  # Click fifth colour (purple #7109AA (170, 9, 113))
                    header = overlay_list[5]
                    draw_color = (170, 9, 113)
                    brush_thickness = 15
                if 800 < x1 < 960:  # Click sixth colour (blue #1240AB (171, 64, 18))
                    header = overlay_list[6]
                    draw_color = (171, 64, 18)
                    brush_thickness = 15
                if 960 < x1 < 1120:  # Click seventh colour (green #00CC00 (0, 204, 0))
                    header = overlay_list[7]
                    draw_color = (0, 204, 0)
                    brush_thickness = 15
                if 1120 < x1 < 1280:  # Click eraser
                    header = overlay_list[8]
                    draw_color = (0, 0, 0)
                    brush_thickness = 50

        # Drawing Mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            print('Drawing mode')
            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = x1, y1
            cv2.line(img, (x_prev, y_prev), (x1, y1), draw_color, brush_thickness)
            cv2.line(img_canvas, (x_prev, y_prev), (x1, y1), draw_color, brush_thickness)
            x_prev, y_prev = x1, y1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    # cur_time = time.time()
    # fps = 1 / (cur_time - prev_time)
    # prev_time = cur_time
    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img[0:126, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5)
    cv2.imshow('Painter', img)
    # cv2.imshow('Canvas', img_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()