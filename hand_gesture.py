import cv2
import mediapipe as mp
import pyautogui
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

prev_x = None
action_time = time.time()
cooldown = 1


def is_one_finger_open(hand_landmarks):
    finger_tips_ids = [8, 12, 16, 20]

    # Check if index finger (ID 8) is up
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        # Check that all other fingers are down
        for tip_id in finger_tips_ids[1:]:  # Skip index finger
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                return False
        return True
    return False


# Function to check if the entire hand is open
def is_hand_open(hand_landmarks):
    finger_tips_ids = [4, 8, 12, 16, 20]

    for tip_id in finger_tips_ids:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            return False
    return True


while True:
    ret, frame = cap.read()

    if not ret:
        print("Ignoring empty camera frame")
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    image.flags.writeable = False
    results = hand.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1])
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])

            if is_one_finger_open(hand_landmarks):
                if prev_x is not None:
                    if x < prev_x - 40:
                        if time.time() - action_time > cooldown:
                            print("Move Left")
                            pyautogui.press('left')
                            action_time = time.time()
                    elif x > prev_x + 50:
                        if time.time() - action_time > cooldown:
                            print("Move Right")
                            pyautogui.press('right')
                            action_time = time.time()

            if is_hand_open(hand_landmarks):
                if time.time() - action_time > cooldown:
                    print("pause/play")
                    pyautogui.press('space')
                    action_time = time.time()

            prev_x = x

    cv2.imshow("Hand Gesture Control", image)

    key = cv2.waitKey(1)
    if key == 27 or key == 13:
        break

cap.release()
cv2.destroyAllWindows()
