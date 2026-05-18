# Handles file paths and directories
import os

# Used for timing, FPS, and latency calculation
import time

# Used for saving logs in JSON format
import json

# Queue structure for detection logs
from collections import deque

# Used for timestamps and filenames
from datetime import datetime

# OpenCV for image and video processing
import cv2

# PyTorch Deep Learning framework
import torch

# Neural network modules
import torch.nn as nn

# Image preprocessing utilities
from torchvision import transforms, models

# YOLO object detection model
from ultralytics import YOLO


# MAIN VISION PIPELINE CLASS
class VisionPipeline:

    # INITIALIZATION FUNCTION
    def __init__(
        self,
        yolo_weights="models/yolo/yolo.pt",
        gesture_ckpt="models/cnn/cnn.pth",
        video_source=0,
        img_size=640,
        person_class=0,
        weapon_classes=None,
        weapon_conf=0.5,
        person_conf=0.5,
        assault_thresh=0.5,
        assault_consec=2,
        cooldown_sec=3.0,
        out_dir="outputs",
        log_keep=50
    ):

        # SELECT DEVICE (GPU OR CPU)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # LOAD YOLO MODEL
        self.yolo = YOLO(yolo_weights)

        # LOAD CNN MODEL CHECKPOINT
        ckpt = torch.load(
            gesture_ckpt,
            map_location=self.device
        )

        # INITIALIZE RESNET18 MODEL
        self.gesture_model = models.resnet18(weights=None)

        # MODIFY OUTPUT LAYER FOR 2 CLASSES
        self.gesture_model.fc = nn.Linear(
            self.gesture_model.fc.in_features,
            2
        )

        # LOAD TRAINED CNN WEIGHTS
        self.gesture_model.load_state_dict(
            ckpt["state_dict"]
        )

        # SET MODEL TO EVALUATION MODE
        self.gesture_model.eval().to(self.device)

        # IMAGE PREPROCESSING PIPELINE
        self.tf = transforms.Compose([

            # Convert image to PIL format
            transforms.ToPILImage(),

            # Resize image
            transforms.Resize((224, 224)),

            # Convert image to tensor
            transforms.ToTensor()
        ])

        # INITIALIZE CAMERA
        self.cap = cv2.VideoCapture(video_source)

        # DEFINE WEAPON CLASSES
        self.weapon_classes = (
            weapon_classes
            or {
                1: "Pistol",
                2: "Knife"
            }
        )

        # DETECTION SETTINGS
        self.weapon_conf = weapon_conf
        self.person_conf = person_conf
        self.person_class = person_class

        self.img_size = img_size
        self.assault_thresh = assault_thresh
        self.assault_consec = assault_consec
        self.cooldown_sec = cooldown_sec

        # ASSAULT DETECTION COUNTER
        self.assault_streak = 0

        # LAST ALERT TIME
        self.last_event_time = 0.0

        # DETECTION LOGS
        self.logs = deque(maxlen=log_keep)

        # INITIAL STATUS
        self.latest_status = {
            "status": "INIT"
        }

        # CURRENT VIDEO FRAME
        self.latest_frame = None

        # OUTPUT DIRECTORIES
        self.out_events = os.path.join(
            out_dir,
            "events"
        )

        self.out_frames = os.path.join(
            out_dir,
            "frames"
        )

        # CREATE OUTPUT FOLDERS
        os.makedirs(
            self.out_events,
            exist_ok=True
        )

        os.makedirs(
            self.out_frames,
            exist_ok=True
        )

        # PERFORMANCE VARIABLES
        self.frame_count = 0

        self.fps_start_time = time.time()

        self.fps = 0.0

        self.latency = 0.0

    # CNN GESTURE PREDICTION FUNCTION
    @torch.no_grad()
    def gesture_predict(self, roi):

        # Convert image from BGR to RGB
        rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        # Apply preprocessing
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        # Predict probabilities
        probs = torch.softmax(
            self.gesture_model(x),
            dim=1
        )[0]

        # Return probabilities
        return (
            float(probs[0]),
            float(probs[1])
        )

    # MAIN PIPELINE PROCESS
    def step(self):

        # Read frame from camera
        ok, frame = self.cap.read()

        # Stop if camera fails
        if not ok:
            return

        # Start latency timer
        t0 = time.time()

        # Copy frame for display
        display = frame.copy()

        # Get frame dimensions
        h, w = frame.shape[:2]

        # RUN YOLO DETECTION
        results = self.yolo(
            frame,
            imgsz=self.img_size,
            verbose=False
        )[0]

        # INITIALIZE VARIABLES
        best_weapon = None

        best_assault_p = 0.0

        weapon_trigger = False

        assault_frame = False

        # LOOP THROUGH DETECTIONS
        for box in results.boxes:

            # Get detected class
            cls = int(box.cls[0])

            # Get confidence score
            conf = float(box.conf[0])

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # PERSON DETECTION + CNN ANALYSIS
            if (
                cls == self.person_class
                and conf >= self.person_conf
            ):

                # Extract person region
                roi = frame[
                    max(0, y1):min(h, y2),
                    max(0, x1):min(w, x2)
                ]

                # Skip empty ROI
                if roi.size == 0:
                    continue

                # Predict gesture
                assault_p, _ = self.gesture_predict(roi)

                # CHECK IF PERSON IS HOLDING WEAPON
                holding_weapon = False

                for wbox in results.boxes:

                    wcls = int(wbox.cls[0])

                    wconf = float(wbox.conf[0])

                    if (
                        wcls in self.weapon_classes
                        and wconf >= self.weapon_conf
                    ):

                        wx1, wy1, wx2, wy2 = map(
                            int,
                            wbox.xyxy[0]
                        )

                        # Check if weapon is inside person box
                        if (
                            wx1 >= x1 and wy1 >= y1
                            and wx2 <= x2 and wy2 <= y2
                        ):

                            holding_weapon = True
                            break

                # Force suspicious if holding weapon
                if holding_weapon:
                    assault_p = 1.0

                # Save best suspicious score
                if assault_p > best_assault_p:
                    best_assault_p = assault_p

                # Check suspicious threshold
                if assault_p >= self.assault_thresh:
                    assault_frame = True

                # Red for suspicious
                # Green for normal
                color = (
                    (0, 0, 255)
                    if assault_p >= self.assault_thresh
                    else (0, 255, 0)
                )

                # Create label
                label = (
                    "SUSPICIOUS BEHAVIOR"
                    if assault_p >= self.assault_thresh
                    else "NORMAL"
                )

                # Draw bounding box
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw label
                cv2.putText(
                    display,
                    f"{label} {assault_p:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # WEAPON DETECTION
            elif (
                cls in self.weapon_classes
                and conf >= self.weapon_conf
            ):

                weapon_trigger = True

                # Get weapon name
                name = self.weapon_classes[cls]

                # Save best weapon
                if (
                    best_weapon is None
                    or conf > best_weapon[1]
                ):

                    best_weapon = (
                        name,
                        conf
                    )

                # Draw weapon box
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                # Draw weapon label
                cv2.putText(
                    display,
                    f"{name} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # ASSAULT STREAK CHECK
        self.assault_streak = (
            self.assault_streak + 1
            if assault_frame
            else 0
        )

        # ALERT THRESHOLD CHECK
        assault_trigger = (
            self.assault_streak >= self.assault_consec
        )

        # DETERMINE SYSTEM STATUS
        if weapon_trigger:
            status = "DANGER"

        elif assault_trigger:
            status = "ALERT"

        else:
            status = "NORMAL"

        # FPS CALCULATION
        self.frame_count += 1

        elapsed = (
            time.time() - self.fps_start_time
        )

        if elapsed >= 1.0:

            self.fps = (
                self.frame_count / elapsed
            )

            self.frame_count = 0

            self.fps_start_time = time.time()

        # LATENCY CALCULATION
        self.latency = (
            time.time() - t0
        ) * 1000

        # UPDATE STATUS DATA
        self.latest_status = {
            "status": status,
            "fps": round(self.fps, 2),
            "latency_ms": round(self.latency, 2),
            "assault_streak": self.assault_streak
        }

        # CHECK IF ALERT EXISTS
        alert = (
            status != "NORMAL"
        )

        now = time.time()

        # SAVE EVENT IF COOLDOWN PASSED
        if (
            alert
            and (
                now - self.last_event_time
            ) >= self.cooldown_sec
        ):

            self.last_event_time = now

            # Generate unique event ID
            eid = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            # Save frame path
            frame_path = os.path.join(
                self.out_frames,
                f"{eid}.jpg"
            )

            # Save JSON path
            json_path = os.path.join(
                self.out_events,
                f"{eid}.json"
            )

            # Save image
            cv2.imwrite(
                frame_path,
                display
            )

            # Create timestamp
            ts = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # CREATE LOG ITEM
            if (
                weapon_trigger
                and best_weapon is not None
            ):

                log_item = {
                    "type": best_weapon[0],
                    "confidence": round(
                        float(best_weapon[1]),
                        2
                    ),
                    "timestamp": ts
                }

            else:

                log_item = {
                    "type": "Suspicious Behavior",
                    "confidence": round(
                        float(best_assault_p),
                        2
                    ),
                    "timestamp": ts
                }

            # Add log to queue
            self.logs.appendleft(log_item)

            # SAVE EVENT METADATA
            meta = {
                "event_id": eid,
                "timestamp": ts,
                "status": status,
                "weapon_trigger": weapon_trigger,
                "assault_trigger": assault_trigger,
                "assault_streak": self.assault_streak,
                "best_assault_prob": round(
                    float(best_assault_p),
                    4
                ),
                "saved_frame": frame_path
            }

            # Save JSON file
            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    meta,
                    f,
                    indent=2
                )

        # ENCODE FRAME TO JPEG
        _, jpg = cv2.imencode(
            ".jpg",
            display
        )

        # SAVE FINAL FRAME
        self.latest_frame = jpg.tobytes()