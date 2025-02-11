import cv2
import numpy as np
import time
import threading
from playsound import playsound

# Load video
cap = cv2.VideoCapture("odd_activity.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the video resolution to Full HD (1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

# Read initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret:
    print("Error: Could not read frames. Exiting...")
    cap.release()
    exit()

# Get screen resolution to fit the video in full screen
screen_width = 1920  # Adjust according to your screen resolution
screen_height = 1080  # Adjust according to your screen resolution

# Create a window for full screen display
cv2.namedWindow("Security Monitoring", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Security Monitoring", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_alert_time = 0
alert_cooldown = 5  # Cooldown time to avoid continuous alerts
alert_playing = False  
motion_history = []  # To track movement over time

def play_alert():
    global alert_playing
    alert_playing = True
    playsound("alert.mp3")
    alert_playing = False

while cap.isOpened():
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Compute absolute difference
    frame_diff = cv2.absdiff(gray1, gray2)

    # Thresholding to highlight movement
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_activity_detected = False
    large_movements = []

    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Adjusted threshold to detect movement
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            large_movements.append(center)

            # Track movement over time
            motion_history.append(center)
            if len(motion_history) > 10:  
                motion_history.pop(0)  # Keep only recent movements

            # Detect sudden movement (attack-like behavior)
            if len(motion_history) >= 5:
                speed = np.linalg.norm(np.array(motion_history[-1]) - np.array(motion_history[0]))
                if speed > 50:  # Sudden fast movement detected
                    suspicious_activity_detected = True
                    cv2.putText(frame1, "ALERT: Attack Detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Play alert sound with cooldown
    current_time = time.time()
    if suspicious_activity_detected and (current_time - last_alert_time > alert_cooldown) and not alert_playing:
        last_alert_time = current_time
        threading.Thread(target=play_alert, daemon=True).start()

    # Resize frame to fit full screen
    frame_resized = cv2.resize(frame1, (screen_width, screen_height))

    cv2.imshow("Security Monitoring", frame_resized)

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

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
