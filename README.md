Driver Drowsiness Detection using Computer Vision

A real-time Driver Sleeping Alert System built using Python, OpenCV, and Deep Learning to detect signs of driver fatigue or drowsiness. The system continuously monitors the driver’s eyes and facial landmarks through the webcam and triggers an alert when signs of sleep are detected.

🧠 Features

Real-time detection using webcam

Eye blink and closed-eye duration tracking

Automatic alert sound when the driver appears drowsy

Lightweight — runs on a single Python file

Customizable alert threshold

🛠️ Technologies Used

Python 3.x

OpenCV — for real-time image processing

dlib / mediapipe — for facial landmark detection

playsound / winsound — for alert sound (optional)

📦 Installation

Clone this repository

git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection


Install dependencies

pip install opencv-python dlib playsound imutils


(If you used Mediapipe instead of Dlib, then run:)

pip install mediapipe


Run the Python script

python driver_alert.py

🎥 How It Works

The webcam captures the driver’s face in real time.

Facial landmarks are detected to locate the eyes.

The system calculates the Eye Aspect Ratio (EAR) or uses blink frequency.

If eyes remain closed beyond a threshold, an alarm is triggered.
