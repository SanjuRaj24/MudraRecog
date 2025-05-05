import os
import cv2
import shutil
import mediapipe as mp

# Gesture configuration
gesture_name = "  "  #Enter the mudra name that you want to create dataset of
num_samples = 100  # Number of images per mudra

# Initialize Mediapipe hands solution
mp_hands = mp.solutions.hands  # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_captured = 0

# Create gesture directory
gesture_dir = f"training/{gesture_name}"
if os.path.exists(gesture_dir):
    shutil.rmtree(gesture_dir, ignore_errors=True)
os.makedirs(gesture_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Initialize bounding box coordinates
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            # Update bounding box based on hand landmarks
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x, landmark.y
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # Convert normalized coordinates to pixel coordinates
            height, width, _ = frame.shape
            x_min = max(0, int(x_min * width) - 30)
            x_max = min(width, int(x_max * width) + 20)
            y_min = max(0, int(y_min * height) - 30)
            y_max = min(height, int(y_max * height) + 20)

            # Crop and save the hand region
            try:
                hand_image = frame[y_min:y_max, x_min:x_max]
                image_filename = os.path.join(gesture_dir, f"image_{frame_captured}.jpg")
                cv2.imwrite(image_filename, hand_image)
                frame_captured += 1

                # Display cropped hand image
                cv2.imshow("Cropped Hand Image", hand_image)
            except Exception as e:
                print(f"Error processing frame: {e}")

    # Display original frame
    cv2.imshow("Frame", frame)

    # Stop capturing when required number of samples is reached
    if (cv2.waitKey(1) & 0xFF == ord("q")) or frame_captured >= num_samples:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
