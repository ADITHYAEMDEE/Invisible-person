# Invisible person

A webcam trick that makes you disappear when you open your hand, using hand tracking and person segmentation with the help of OpenCV and mediapipe.

## How it works:
Capture a clean background image of your room (no you in it).
Track your hand with MediaPipe to detect open vs. closed.
Opening your hand toggles "invisible" mode on/off.
While invisible, the area around your body is replaced with the background image, leaving a soft-edged cutout with a purple outline.
A sound loops while you're invisible, and a different sound plays once when you reappear.

## Requirements:
Python 3
opencv-python
mediapipe
numpy
pygame

### You'll also need these files in the same folder as the script:
hand_landmarker.task
selfie_segmenter.tflite
dis.wav (plays on loop while invisible)
app.wav (plays once when you reappear)

## Usage:
1. Run the script: python main.py
2. Step out of frame and press b to capture the background.
3. Step back in.
4. Open your hand to vanish, open it again to reappear.
5. Press q to quit.
