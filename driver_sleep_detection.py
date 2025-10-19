import cv2
import mediapipe as mp
from playsound import playsound
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25        # Eye aspect ratio threshold
EAR_CONSEC_FRAMES = 20      # Consecutive frames to consider sleepy
COUNTER = 0

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * img_w)
        y = int(landmarks[idx].y * img_h)
        points.append((x, y))
    # Vertical distances
    A = ((points[1][0]-points[5][0])**2 + (points[1][1]-points[5][1])**2)**0.5
    B = ((points[2][0]-points[4][0])**2 + (points[2][1]-points[4][1])**2)**0.5
    # Horizontal distance
    C = ((points[0][0]-points[3][0])**2 + (points[0][1]-points[3][1])**2)**0.5
    ear = (A + B) / (2.0 * C)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            # Check for drowsiness
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "SLEEP ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    try:
                        playsound("alarm.mp3")
                    except:
                        pass
            else:
                COUNTER = 0

            # Display EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
