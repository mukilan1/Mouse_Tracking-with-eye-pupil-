# Mouse Tracking with Eye Pupil

This project demonstrates real-time eye tracking using computer vision techniques to control the mouse cursor based on the movement of eye pupils. It utilizes the MediaPipe library for facial landmark detection and OpenCV for video processing. The mouse cursor is controlled by tracking the average position of both left and right eye pupils, providing accurate and responsive control suitable for users with disabilities.

## Features

- Real-time detection of eye pupils from a video feed.
- Mapping of pupil movements to control the mouse cursor.
- Display of eye and pupil landmarks on the video feed for visual feedback.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI (for mouse cursor control)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mouse-tracking-eye-pupil.git
   cd mouse-tracking-eye-pupil
   pip install -r requirements.py
