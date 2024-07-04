import cv2
import mediapipe as mp
import pyautogui

# Function to move the mouse cursor based on the average pupil positions
def move_mouse(avg_left_x, avg_left_y, avg_right_x, avg_right_y):
    screen_width, screen_height = pyautogui.size()
    avg_x = (avg_left_x + avg_right_x) / 2
    avg_y = (avg_left_y + avg_right_y) / 2
    mouse_x = int(avg_x * screen_width)
    mouse_y = int(avg_y * screen_height)
    pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

def detect_pupils_and_control_mouse(video_path):
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize MediaPipe Face Mesh.
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    # Indices for the eye landmarks (including irises)
    left_eye_landmarks = [33, 133, 144, 145, 153, 154, 155, 159, 160, 161, 163, 173]
    right_eye_landmarks = [362, 382, 383, 384, 385, 386, 387, 388, 390, 398]
    left_iris_landmarks = [468, 469, 470, 471]
    right_iris_landmarks = [473, 474, 475, 476]

    frame_count = 0  # Counter to track frames for processing

    while cap.isOpened():
        # Read every nth frame (e.g., every 3rd frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue  # Skip frames to improve processing speed

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and detect face mesh landmarks.
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks for the left and right irises
                left_iris_points = []
                right_iris_points = []
                for idx in left_iris_landmarks:
                    left_iris_points.append((face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y))
                for idx in right_iris_landmarks:
                    right_iris_points.append((face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y))

                # Calculate average position of left iris (pupil)
                if left_iris_points:
                    avg_left_x = sum([x for x, y in left_iris_points]) / len(left_iris_points)
                    avg_left_y = sum([y for x, y in left_iris_points]) / len(left_iris_points)

                # Calculate average position of right iris (pupil)
                if right_iris_points:
                    avg_right_x = sum([x for x, y in right_iris_points]) / len(right_iris_points)
                    avg_right_y = sum([y for x, y in right_iris_points]) / len(right_iris_points)

                # Move the mouse cursor based on average pupil positions
                move_mouse(avg_left_x, avg_left_y, avg_right_x, avg_right_y)

                # Draw landmarks for the left eye and iris
                for idx in left_eye_landmarks:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                for idx in left_iris_landmarks:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                # Draw landmarks for the right eye and iris
                for idx in right_eye_landmarks:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                for idx in right_iris_landmarks:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Display the frame.
        cv2.imshow('Eye and Pupil Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'EyeMovement.mp4'  # Replace with your video file path
    detect_pupils_and_control_mouse(video_path)
