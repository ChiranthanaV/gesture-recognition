import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Open webcam (try 0 or 1 depending on your setup)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

def get_gesture(landmarks):
    def is_finger_up(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y

    # Finger landmark indices
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    # Finger states
    thumb_up = landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y
    thumb_down = landmarks[THUMB_TIP].y > landmarks[THUMB_IP].y and landmarks[THUMB_TIP].y > landmarks[THUMB_MCP].y
    index_up = is_finger_up(INDEX_TIP, INDEX_PIP)
    middle_up = is_finger_up(MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_up(RING_TIP, RING_PIP)
    pinky_up = is_finger_up(PINKY_TIP, PINKY_PIP)

    # 🎯 Gesture Logic
    if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
        return "Thumbs Up!"
    elif thumb_down and not (index_up or middle_up or ring_up or pinky_up):
        return "Thumbs Down!"
    elif index_up and middle_up and not (ring_up or pinky_up or thumb_up):
        return "Peace!"
    elif all([thumb_up, index_up, middle_up, ring_up, pinky_up]):
        return "Palm!"
    elif index_up and pinky_up and not (middle_up or ring_up or thumb_up):
        return "Rock!"
    else:
        return "Unrecognized"


while True:
    success, img = cap.read()

    if not success or img is None:
        print("⚠️ Warning: Empty frame received.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks.landmark)
            cv2.putText(img, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
