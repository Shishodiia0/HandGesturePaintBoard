import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *

# Initialize Pygame window for drawing
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Drawing App")

# Set background color
background_color = (255, 255, 255)
screen.fill(background_color)

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to get hand landmarks and draw them
def draw_hand_landmarks(frame, hands_landmarks):
    if hands_landmarks:
        for hand_landmarks in hands_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

# Function to handle drawing
def handle_drawing(frame, hand_landmarks, drawing=True, color=(0, 0, 0), brush_size=5):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            # Get the tip of the index finger (landmark 8)
            finger_tip = hand_landmark.landmark[8]
            x, y = int(finger_tip.x * WIDTH), int(finger_tip.y * HEIGHT)

            # If drawing, draw a circle
            if drawing:
                pygame.draw.circle(screen, color, (x, y), brush_size)

            # Update Pygame display
            pygame.display.update()

# Function to change brush size and color based on gestures
def change_brush_and_color(frame, hand_landmarks):
    brush_size = 5
    color = (0, 0, 0)  # Default color (Black)

    if hand_landmarks:
        # Get the index finger and thumb distance (for size control)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        dist = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))

        # Map the distance to a brush size range
        brush_size = int(dist * 10)

        # Color change based on finger positions or gesture (e.g., thumb up for color change)
        if dist > 0.1:  # example threshold for color change gesture
            color = (255, 0, 0)  # Red color for example
        else:
            color = (0, 0, 0)  # Black color for default
        
    return brush_size, color

# Main program loop
cap = cv2.VideoCapture(0)

drawing = False
brush_size = 5
color = (0, 0, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand gestures
    result = hands.process(frame_rgb)
    
    # Draw the hand landmarks
    frame = draw_hand_landmarks(frame, result.multi_hand_landmarks)
    
    # Get brush size and color based on gestures
    brush_size, color = change_brush_and_color(frame, result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None)
    
    # Handle drawing on the screen
    handle_drawing(frame, result.multi_hand_landmarks, drawing=drawing, color=color, brush_size=brush_size)

    # Show the webcam feed (for debugging)
    cv2.imshow('Webcam Feed', frame)

    # Detect key press (for quitting)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cv2.destroyAllWindows()
            cap.release()
            exit()
        if event.type == KEYDOWN:
            if event.key == K_c:  # Press 'C' to clear screen
                screen.fill(background_color)
                pygame.display.update()

    # Update the Pygame window
    pygame.display.update()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
