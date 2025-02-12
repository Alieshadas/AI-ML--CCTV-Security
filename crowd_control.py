import cv2
import numpy as np
import pygame
import time
import threading
import paho.mqtt.client as mqtt

# ✅ Initialize Pygame for sound
pygame.mixer.init()
alarm_sound = "alarm.mp3"

# ✅ Load MobileNet SSD model
prototxt_path = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model_path = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ✅ MQTT Setup
MQTT_BROKER = "test.mosquitto.org"  # Public MQTT Broker
MQTT_TOPIC = "crowd/alert"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)

# ✅ IP camera stream
video_url = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

CONFIDENCE_THRESHOLD = 0.4
CROWD_THRESHOLD = 3 # Change as per requirement
last_alert_time = 0
alert_cooldown = 5  # 5 seconds cooldown
alarm_playing = False

frame = None
fgbg = cv2.createBackgroundSubtractorMOG2()
program_running = True

def capture_frame():
    global frame
    while program_running:
        ret, new_frame = cap.read()
        if ret:
            frame = new_frame
        else:
            break

def process_frames():
    global frame, last_alert_time, alarm_playing
    while program_running:
        if frame is None:
            continue
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_count = sum(1 for i in range(detections.shape[2]) if detections[0, 0, i, 2] > CONFIDENCE_THRESHOLD and int(detections[0, 0, i, 1]) == 15)

        cv2.putText(frame, f"Crowd: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        current_time = time.time()
        
        if person_count >= CROWD_THRESHOLD:
            cv2.putText(frame, "HIGH ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            print("🚨 ALERT: High Crowd Density Detected! 🚨")
            
            if not alarm_playing and (current_time - last_alert_time > alert_cooldown):
                pygame.mixer.music.load(alarm_sound)
                pygame.mixer.music.play()
                last_alert_time = current_time
                alarm_playing = True  

                # ✅ Send MQTT Alert
                mqtt_client.publish(MQTT_TOPIC, "ALERT: Crowd Limit Exceeded!")  

        else:
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False  

        cv2.imshow("Crowd Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_program()

def stop_program():
    global program_running
    program_running = False
    print("Exiting program...")

# ✅ Start threads
threading.Thread(target=capture_frame, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

while program_running:
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
