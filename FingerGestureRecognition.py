import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

distances = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

pTime = 0
cTime = 0

gesture_map = {0: "Fist", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    counter = 0

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, landmark, mp_hands.HAND_CONNECTIONS)
            motions = []

            for id, lm in enumerate(landmark.landmark):
                h, w, _ = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                motions.append([id, x, y])

            for item in distances:
                downFingerY = motions[item[0]][2]
                upperFingerY = motions[item[1]][2]
                isFingerOpen = downFingerY > upperFingerY
                counter += 1 if isFingerOpen else 0

            gesture = gesture_map.get(counter, "Unknown")
            cv2.putText(img, f'Gesture: {gesture}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
