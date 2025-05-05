# MudraRecog 🤲 | AI-Based Hand Mudra Recognition System

**MudraRecog** is a real-time hand gesture recognition system that detects and classifies various **Indian mudras** using computer vision and deep learning. It leverages **MediaPipe** for hand landmark detection and a fine-tuned **MobileNetV2** model for accurate mudra classification. This project was designed with accessibility, real-world usability, and user interaction in mind.

## 🚀 Project Highlights

- Real-time hand tracking using **MediaPipe**
- Transfer learning with **MobileNetV2**
- Highly accurate mudra recognition with confidence scores close to 1.00
- Automatic dataset collection via webcam
- Organized project structure with modular code and clean outputs
- Lightweight `.keras` model for efficient deployment

---

## 📂 Folder Structure

MudraRecog/
│
├── code/ # Scripts for dataset collection, training, and prediction
├── model/ # Trained model files 
├── outputs/ # Screenshots or visual results of predictions
├── class_indices.json # Class-label mapping used during training and inference
└── README.md # Project documentation


## ⚙️ How It Works

1. Dataset Collection
   - Run the dataset script to collect mudra images via webcam.
   - Make sure to update the mudra name in the script before each run.
   - Each image is cropped and saved automatically based on detected hand landmarks.

2. Model Training
   - Uses **MobileNetV2** as a base model with custom classification layers.
   - Two-phase training:
     - Initial training with frozen base layers.
     - Fine-tuning the last few layers for enhanced accuracy.
   - Trained on a structured folder-based dataset (training/validation).

3. Real-Time Prediction
   - The model predicts mudras live from webcam input.
   - Uses the `class_indices.json` file to map predictions to mudra names.

## 🛠️ Tech Stack

- **TensorFlow** & **Keras** – Model building and training
- **MediaPipe** – Real-time hand tracking
- **OpenCV** – Webcam access and image handling
- **NumPy** – Data processing
- **Python** – Core programming language

## 📸 Sample Output

A few visual results from real-time mudra recognition can be found in the `outputs/` folder.

## 📌 Usage Instructions

> You can run the system using **CMD or Visual Studio Code**:
python filename.py

➕ To collect data for a new mudra:
Open dataset.py in the code/ folder.

Change the value of gesture_name = "YourMudraName" in the script.

Run the script and perform the gesture in front of the webcam.

The images will be saved under dataset/training/<YourMudraName>/.

📁 Important File
class_indices.json
Maps numeric class indices to actual mudra names. This is essential for interpreting prediction results and should always be kept updated when adding new gestures.

💡 Future Improvements
Add GUI for easier dataset collection and prediction
Expand dataset for more mudras and multi-user support


👨‍💻 Author
Created with dedication and passion as a final year project in B.Voc Software Development.

📬 Contact
For queries, feedback, or collaboration opportunities, feel free to reach out via GitHub.
