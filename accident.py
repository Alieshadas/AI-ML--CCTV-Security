import cv2
import numpy as np
import time
import threading
import os
import pygame  # Switched from playsound to pygame for better sound handling

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

# Load video (replace with real-time feed if needed)
cap = cv2.VideoCapture("accident.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if not detected
frame_delay = int(1000 / (fps * 1.5))  # Adjusted speed to 1.5x

# Read first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret:
    print("Error: Could not read frames. Exiting...")
    cap.release()
    exit()

# Alert variables
last_alert_time = 0
alert_cooldown = 5  # Alert cooldown in seconds
alert_playing = False  
motion_history = []

# Sound file path
sound_path = os.path.abspath("sound.mp3")

# Function to play alert sound
def play_alert():
    global alert_playing
    alert_playing = True
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until sound finishes
            time.sleep(0.1)
    except Exception as e:
        print("Error playing sound:", e)
    alert_playing = False

while cap.isOpened():
    # Convert frames to grayscale & apply blur
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Compute frame difference
    frame_diff = cv2.absdiff(gray1, gray2)

    # Thresholding to highlight movement
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accident_detected = False
    impact_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 8000:  # Adjusted threshold for better accuracy
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # Track movement over time
            motion_history.append(center)
            if len(motion_history) > 10:
                motion_history.pop(0)

            # Detect sudden impact based on movement speed
            if len(motion_history) >= 5:
                speed = np.linalg.norm(np.array(motion_history[-1]) - np.array(motion_history[0]))
                if speed > 30:  # Speed threshold for detecting impact
                    accident_detected = True
                    impact_detected = True
                    cv2.putText(frame1, "ALERT: Accident Detected!", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw bounding box around detected movement
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Play alert sound only when an accident is detected
    current_time = time.time()
    if impact_detected and (current_time - last_alert_time > alert_cooldown) and not alert_playing:
        last_alert_time = current_time
        threading.Thread(target=play_alert, daemon=True).start()

    # Show video in fullscreen mode
    cv2.namedWindow("Accident Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Accident Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Accident Detection", frame1)

    # Read next frame
    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        print("Video Ended. Restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Could not restart video.")
            break

    # Exit on 'q' key press
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
