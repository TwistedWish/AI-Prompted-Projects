import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

# UI button positions and sizes
BUTTON_HEIGHT = 60
BUTTON_WIDTH = 160
BUTTON_MARGIN = 20

def count_extended_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    thumb_extended = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    return count, thumb_extended

def draw_buttons(display, mode):
    h, w = display.shape[:2]
    pencil_rect = (BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
    eraser_rect = (BUTTON_MARGIN*2 + BUTTON_WIDTH, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
    clear_rect = (BUTTON_MARGIN*3 + BUTTON_WIDTH*2, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
    cv2.rectangle(display, (pencil_rect[0], pencil_rect[1]), 
                  (pencil_rect[0]+pencil_rect[2], pencil_rect[1]+pencil_rect[3]), 
                  (0,255,0) if mode=="pencil" else (100,100,100), -1)
    cv2.putText(display, "Pencil", (pencil_rect[0]+20, pencil_rect[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.rectangle(display, (eraser_rect[0], eraser_rect[1]), 
                  (eraser_rect[0]+eraser_rect[2], eraser_rect[1]+eraser_rect[3]), 
                  (0,0,255) if mode=="eraser" else (100,100,100), -1)
    cv2.putText(display, "Eraser", (eraser_rect[0]+20, eraser_rect[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.rectangle(display, (clear_rect[0], clear_rect[1]), 
                  (clear_rect[0]+clear_rect[2], clear_rect[1]+clear_rect[3]), 
                  (255,255,0), -1)
    cv2.putText(display, "Clear", (clear_rect[0]+30, clear_rect[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return pencil_rect, eraser_rect, clear_rect

def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx+rw and ry <= y <= ry+rh

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    # Compute EAR for one eye using 6 points
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    # vertical distances
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    # horizontal distance
    C = np.linalg.norm(p[0] - p[3])
    ear = (A + B) / (2.0 * C)
    return ear

def draw_face_skeleton(display, face_landmarks, w, h):
    # Draw connections for a simple face skeleton (jaw, eyes, nose, mouth)
    # Jaw
    jaw_indices = list(range(0, 17))
    for i in range(len(jaw_indices)-1):
        pt1 = face_landmarks[jaw_indices[i]]
        pt2 = face_landmarks[jaw_indices[i+1]]
        p1 = (int(pt1.x * w), int(pt1.y * h))
        p2 = (int(pt2.x * w), int(pt2.y * h))
        cv2.line(display, p1, p2, (255, 255, 0), 2)
    # Eyes
    left_eye = [33, 160, 158, 133, 153, 144, 163, 7, 246, 161, 159, 27, 23, 130, 243, 112]
    right_eye = [362, 385, 387, 263, 373, 380, 390, 249, 466, 388, 386, 259, 255, 339, 463, 342]
    for eye in [left_eye, right_eye]:
        for i in range(len(eye)):
            pt1 = face_landmarks[eye[i]]
            pt2 = face_landmarks[eye[(i+1)%len(eye)]]
            p1 = (int(pt1.x * w), int(pt1.y * h))
            p2 = (int(pt2.x * w), int(pt2.y * h))
            cv2.line(display, p1, p2, (0, 255, 255), 2)
    # Nose bridge
    for i in [168, 6, 197, 195, 5]:
        if i != 5:
            pt1 = face_landmarks[i]
            pt2 = face_landmarks[i+1]
            p1 = (int(pt1.x * w), int(pt1.y * h))
            p2 = (int(pt2.x * w), int(pt2.y * h))
            cv2.line(display, p1, p2, (255, 0, 255), 2)
    # Mouth
    mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    for i in range(len(mouth)):
        pt1 = face_landmarks[mouth[i]]
        pt2 = face_landmarks[mouth[(i+1)%len(mouth)]]
        p1 = (int(pt1.x * w), int(pt1.y * h))
        p2 = (int(pt2.x * w), int(pt2.y * h))
        cv2.line(display, p1, p2, (0, 128, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    drawing_points = []
    drawing = False
    prev_index_tip = None
    mode = "pencil"
    eraser_radius = 30

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands, mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            status = "Paused"
            index_tip = None
            eye_status = "Unknown"

            # Draw UI buttons
            pencil_rect, eraser_rect, clear_rect = draw_buttons(frame, mode)

            # Face skeleton and eye status
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                draw_face_skeleton(frame, face_landmarks, w, h)
                # EAR for both eyes
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_ear = eye_aspect_ratio(face_landmarks, left_eye_indices, w, h)
                right_ear = eye_aspect_ratio(face_landmarks, right_eye_indices, w, h)
                avg_ear = (left_ear + right_ear) / 2
                if avg_ear < 0.22:
                    eye_status = "Eyes Closed"
                else:
                    eye_status = "Eyes Open"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                finger_count, thumb_extended = count_extended_fingers(hand_landmarks)
                index_up = (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y)
                others_down = all(
                    hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                    for tip, pip in zip([12, 16, 20], [10, 14, 18])
                )
                ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                index_tip = (ix, iy)

                # UI interaction: select mode or clear
                if index_up and others_down:
                    if point_in_rect(index_tip, pencil_rect):
                        mode = "pencil"
                    elif point_in_rect(index_tip, eraser_rect):
                        mode = "eraser"
                    elif point_in_rect(index_tip, clear_rect):
                        drawing_points = []
                        mode = "pencil"
                    elif mode == "pencil":
                        status = "Drawing"
                        drawing = True
                    elif mode == "eraser":
                        status = "Erasing"
                        drawing = True
                    else:
                        drawing = False
                else:
                    drawing = False

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Drawing/Erasing logic
            if drawing and index_tip is not None:
                if mode == "pencil":
                    if prev_index_tip is not None:
                        drawing_points.append((prev_index_tip, index_tip))
                    prev_index_tip = index_tip
                elif mode == "eraser":
                    drawing_points = [
                        (pt1, pt2) for pt1, pt2 in drawing_points
                        if (np.linalg.norm(np.array(pt1)-np.array(index_tip)) > eraser_radius and
                            np.linalg.norm(np.array(pt2)-np.array(index_tip)) > eraser_radius)
                    ]
                    prev_index_tip = None
            else:
                prev_index_tip = None

            # Draw lines on canvas
            for pt1, pt2 in drawing_points:
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 6)
            if mode == "eraser" and index_tip is not None:
                cv2.circle(canvas, index_tip, eraser_radius, (0,0,255), 2)

            # Overlay canvas
            display = cv2.addWeighted(frame, 1, canvas, 1, 0.5)
            cv2.putText(display, f"Status: {mode.capitalize()}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255) if mode == "pencil" else (0, 0, 255), 2)
            cv2.putText(display, f"Eyes: {eye_status}", (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 0) if eye_status == "Eyes Open" else (0, 0, 255), 2)

            cv2.imshow("Finger Drawer", display)
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("Finger Drawer", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()