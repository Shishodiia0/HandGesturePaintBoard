import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
from datetime import datetime

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Points for different colors
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# Index for deque
bindex = 0
gindex = 0
rindex = 0
yindex = 0

# Color palette
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create paint window
paintWindow = np.zeros((471, 700, 3)) + 255  # Extended width for SAVE button

# Draw buttons on paint window
def draw_buttons(window):
    cv2.rectangle(window, (40, 1), (140, 65), (255, 255, 255), -1)   # Clear All
    cv2.rectangle(window, (160, 1), (255, 65), colors[0], -1)       # Blue
    cv2.rectangle(window, (275, 1), (370, 65), colors[1], -1)       # Green
    cv2.rectangle(window, (390, 1), (485, 65), colors[2], -1)       # Red
    cv2.rectangle(window, (505, 1), (600, 65), colors[3], -1)       # Yellow
    cv2.rectangle(window, (620, 1), (695, 65), (200, 200, 200), -1) # Save button

    cv2.putText(window, "CLEAR", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(window, "BLUE", (185, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(window, "GREEN", (298, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(window, "RED", (420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(window, "YELLOW", (515, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    cv2.putText(window, "SAVE", (635, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

# Initialize buttons
draw_buttons(paintWindow)
cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

# Initialize camera
camera = cv2.VideoCapture(0)

# Function to save the painting
def save_painting():
    save_dir = "drawings"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"painting_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, paintWindow[67:, :])
    print(f"Painting saved as '{filepath}'.")

# Function to check if hand is making a fist
def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    return all([index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y])

while True:
    ret, frame = camera.read()
    if not ret:
        print("Camera feed not available.")
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to match extended paint window
    frame = cv2.resize(frame, (700, 471))

    # Add buttons to the live frame
    draw_buttons(frame)

    result = hands.process(frameRGB)
    indexFingerTip = None
    drawing = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            indexFingerTip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            indexFingerTip = (int(indexFingerTip.x * w), int(indexFingerTip.y * h))

            if is_fist(hand_landmarks):
                drawing = False
            else:
                drawing = True

            if indexFingerTip[1] <= 65:
                if 40 <= indexFingerTip[0] <= 140:  # Clear
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    paintWindow[67:, :] = 255
                elif 160 <= indexFingerTip[0] <= 255:
                    colorIndex = 0
                elif 275 <= indexFingerTip[0] <= 370:
                    colorIndex = 1
                elif 390 <= indexFingerTip[0] <= 485:
                    colorIndex = 2
                elif 505 <= indexFingerTip[0] <= 600:
                    colorIndex = 3
                elif 620 <= indexFingerTip[0] <= 695:
                    save_painting()
            else:
                if drawing:
                    if colorIndex == 0:
                        bpoints[bindex].appendleft(indexFingerTip)
                    elif colorIndex == 1:
                        gpoints[gindex].appendleft(indexFingerTip)
                    elif colorIndex == 2:
                        rpoints[rindex].appendleft(indexFingerTip)
                    elif colorIndex == 3:
                        ypoints[yindex].appendleft(indexFingerTip)

    for points, color in zip([bpoints, gpoints, rpoints, ypoints], colors):
        for line in points:
            for i in range(1, len(line)):
                if line[i - 1] is None or line[i] is None:
                    continue
                cv2.line(frame, line[i - 1], line[i], color, 2)
                cv2.line(paintWindow, line[i - 1], line[i], color, 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        save_painting()

camera.release()
cv2.destroyAllWindows()
