# main.py
import os
import sys
import time
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from servo_control import move_tilting, rotate_waste, neutral_position, cleanup

# ------------------ USER PARAMETERS ------------------
model_path = "v5_ncnn_model"   # NCNN model folder
usb_idx = 0
min_thresh = 0.5
resW, resH = 640, 480
# ------------------------------------------------------

# Model folder check
if not os.path.exists(model_path):
    print("ERROR: Model folder not found")
    sys.exit(0)

# Load YOLO-NCNN
model = YOLO(model_path, task="detect")
labels = model.names

# Init camera
cap = cv2.VideoCapture(usb_idx)
cap.set(3, resW)
cap.set(4, resH)

# Move servos to neutral on startup
neutral_position()

votes = []
frame_rate_buffer = []
fps_avg_len = 200

try:
    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera failure — exiting.")
            break

        frame = cv2.resize(frame, (resW, resH))

        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Collect classification votes
        for i in range(len(detections)):
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            votes.append(classname)

        # ---------------- OPTION B: Process every 10 votes ----------------
        if len(votes) >= 10:
            counts = Counter(votes)
            final_class = counts.most_common(1)[0][0]

            print(f"Final decision after 10 cycles → {final_class}")

            # ---- Servo Actions ----
            if final_class == "Paper":
                move_tilting(38, 132)

            elif final_class == "Plastic":
                move_tilting(132, 38)

            elif final_class in ["Metal", "Decomposable"]:
                rotate_waste(165)
                if final_class == "Metal":
                    move_tilting(38, 132)
                else:
                    move_tilting(132, 38)

            # Back to neutral
            time.sleep(2)
            neutral_position()

            # Reset for next cycle
            votes.clear()
            print("Ready for next object...\n")
            time.sleep(1)

        # FPS Calculation
        t_stop = time.perf_counter()
        fps = 1 / (t_stop - t_start)
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(fps)

except KeyboardInterrupt:
    print("Manual exit detected")

finally:
    cap.release()
    cleanup()
    print(f"Average FPS: {np.mean(frame_rate_buffer):.2f}")
