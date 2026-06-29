Blind Voice Assistant – Real-Time Object Detection
This project uses YOLOv2, OpenCV, and Text-to-Speech to detect objects in real time and announce their movement direction (left or right), assisting visually impaired users.

Features
Real-time object detection using webcam
Voice alerts using pyttsx3
Detects movement direction (left / right)
Supports 80+ COCO dataset classes
Press Q to exit
Requirements
pip install opencv-python numpy pyttsx3

Files Needed
blind_voice_assistant.py
yolov2.cfg
yolov2.weights
coco.names
(All files must be in the same directory.)

How to Run
python blind_voice_assistant.py

How It Works
Captures video from webcam
Runs YOLOv2 object detection
Tracks object positions across frames
Speaks object name + movement direction
Displays bounding boxes on screen
License
MIT License
