import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Load trained model
model = tf.keras.models.load_model("mudra_mobilenetv2_finetuned.keras")
print("Model loaded successfully.")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse dictionary to get class labels
class_labels = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
img_size = 224 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from webcam!")
        break

    # Convert to RGB (MediaPipe requires RGB input)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # hand bounding box
            h, w, _ = frame.shape 
            x_min, y_min, x_max, y_max = w, h, 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Expand bounding box 
            margin = 20
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size == 0:
                continue

            # Preprocess image for model
            hand_img = cv2.resize(hand_img, (img_size, img_size))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict
            pred = model.predict(hand_img)[0]
            predicted_index = np.argmax(pred)
            confidence = pred[predicted_index]
            label = class_labels.get(predicted_index, "Unknown")

            # Display prediction
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "No Hand Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Mudra Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
