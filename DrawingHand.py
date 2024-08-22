import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

draw_color = (255, 255, 255)
erase_color = (0, 0, 0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = None, None

def draw_line(canvas, start, end, color, thickness=2):
	cv2.line(canvas, start, end, color, thickness)

def erase_area(canvas, center, radius, color):
	cv2.circle(canvas, center, radius, color, -1)

while True:
	success, img = cap.read()
	if not success:
		break
	img = cv2.flip(img, 1)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(img_rgb)

	if results.multi_hand_landmarks:
		for landmark in results.multi_hand_landmarks:
			landmarks = landmark.landmark
			indexTipX, indexTipY = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1]), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])
			palmX, palmY = int(landmarks[mp_hands.HandLandmark.WRIST].x * img.shape[1]), int(landmarks[mp_hands.HandLandmark.WRIST].y * img.shape[0])
			if results.multi_handedness[0].classification[0].label == 'Left' and landmarks[mp_hands.HandLandmark.WRIST].x < 0.5:
				erase_area(canvas, (palmX, palmY), 50, erase_color)
			else:
				if prev_x is not None and prev_y is not None:
					draw_line(canvas, (prev_x, prev_y), (indexTipX, indexTipY), draw_color, thickness=4)
				prev_x, prev_y = indexTipX, indexTipY
	else:
		prev_x, prev_y = None, None

	img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
	
	cv2.imshow("Image", img)
	cv2.imshow("Canvas", canvas)
	if cv2.waitKey(1) and 0xFF ==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()





