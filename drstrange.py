import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Smooth cursor movement variables
prev_x, prev_y = 0, 0
smooth_factor = 0.4  # Increase for smoother motion

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip & thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert to screen coordinates
            finger_x, finger_y = int(index_finger_tip.x * screen_w), int(index_finger_tip.y * screen_h)

            # Apply smoothing for smoother cursor movement
            smooth_x = int(prev_x + smooth_factor * (finger_x - prev_x))
            smooth_y = int(prev_y + smooth_factor * (finger_y - prev_y))
            prev_x, prev_y = smooth_x, smooth_y

            # Move the cursor
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

            # Measure distance between index finger & thumb for clicking
            finger_distance = np.linalg.norm(
                np.array([index_finger_tip.x, index_finger_tip.y]) -
                np.array([thumb_tip.x, thumb_tip.y])
            )

            # Click if fingers come close
            if finger_distance < 0.05:  # Adjust threshold if needed
                pyautogui.click()
                pyautogui.sleep(0.2)  # Prevent multiple rapid clicks

    # Show Camera Feed
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
