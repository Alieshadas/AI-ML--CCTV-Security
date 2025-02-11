import cv2
import time
import pygame
import threading

# Load Pretrained Cascade for Weapon Detection (Ensure it is weapon-specific cascade)
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Initialize Pygame for Sound
pygame.mixer.init()
alert_sound = "sound.mp3"

prev_time = 0
alert_cooldown = 2  # Minimum time gap between sounds (in seconds)
frame_time = 1 / 30  # Target FPS (30 FPS)
fps_multiplier = 1.5  # Increase FPS multiplier to speed up video
frame_skip = 2  # Skip every 2nd frame to speed up processing (can be adjusted)
frame_counter = 0

def play_alert():
    """ Function to play alert sound without overlapping """
    if not pygame.mixer.get_busy():  # Play only if no other sound is playing
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()

# Start continuous monitoring
while True:
    # Load Video File
    camera = cv2.VideoCapture('test_video.mp4')
    
    # Ensure the video loaded properly
    if not camera.isOpened():
        print("Error: Couldn't open video.")
        break
    
    while True:
        start_time = time.time()  # Start time for FPS control

        ret, frame = camera.read()
        if not ret:
            print("Video ended, restarting...")
            break  # Restart video if ended

        if frame_counter % frame_skip != 0:  # Skip frames to reduce workload
            frame_counter += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optimizing cascade detector to speed up detection
        guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        weapon_detected = False
        for (x, y, w, h) in guns:
            # Only draw rectangle around detected weapon
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            weapon_detected = True

        curr_time = time.time()

        # Play sound only if weapon is detected and previous sound is finished
        if weapon_detected and (curr_time - prev_time) > alert_cooldown:
            print("🔴 Weapon Detected! Playing Alert Sound.")
            threading.Thread(target=play_alert, daemon=True).start()  # Run in a separate thread
            prev_time = curr_time

        # Make the window fullscreen
        cv2.namedWindow("Weapon Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Weapon Detection", cv2.WND_PROP_FULLSCREEN, 1)

        # Display the frame in fullscreen
        cv2.imshow("Weapon Detection", frame)

        # Maintain faster FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time / fps_multiplier - elapsed_time)
        time.sleep(sleep_time)  # Delay to maintain faster FPS

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Program Exiting...")
            camera.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            exit()

        frame_counter += 1

    camera.release()
    cv2.destroyAllWindows()
