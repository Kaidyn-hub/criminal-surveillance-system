import os
import json
import time
from collections import deque
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO


class VisionPipeline:
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
        assault_thresh=0.8,
        assault_consec=5,
        cooldown_sec=3.0,
        out_dir="outputs",
        log_keep=50
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.yolo = YOLO(yolo_weights)

        ckpt = torch.load(
            gesture_ckpt,
            map_location=self.device
        )

        self.gesture_model = models.resnet18(weights=None)

        self.gesture_model.fc = nn.Linear(
            self.gesture_model.fc.in_features,
            2
        )

        self.gesture_model.load_state_dict(
            ckpt["state_dict"]
        )

        self.gesture_model.eval().to(self.device)

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.cap = cv2.VideoCapture(video_source)

        self.weapon_classes = (
            weapon_classes
            or {
                1: "Pistol",
                2: "Knife"
            }
        )

        self.weapon_conf = weapon_conf
        self.person_conf = person_conf
        self.person_class = person_class

        self.img_size = img_size
        self.assault_thresh = assault_thresh
        self.assault_consec = assault_consec
        self.cooldown_sec = cooldown_sec

        self.assault_streak = 0
        self.last_event_time = 0.0

        self.logs = deque(maxlen=log_keep)

        self.latest_status = {
            "status": "INIT"
        }

        self.latest_frame = None

        self.out_events = os.path.join(
            out_dir,
            "events"
        )

        self.out_frames = os.path.join(
            out_dir,
            "frames"
        )

        os.makedirs(
            self.out_events,
            exist_ok=True
        )

        os.makedirs(
            self.out_frames,
            exist_ok=True
        )

        self.frame_count = 0

        self.fps_start_time = time.time()

        self.fps = 0.0

        self.latency = 0.0

    @torch.no_grad()
    def gesture_predict(self, roi):

        rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        x = self.tf(rgb).unsqueeze(0).to(self.device)

        probs = torch.softmax(
            self.gesture_model(x),
            dim=1
        )[0]

        return (
            float(probs[0]),
            float(probs[1])
        )

    def step(self):

        ok, frame = self.cap.read()

        if not ok:
            return

        t0 = time.time()

        display = frame.copy()

        h, w = frame.shape[:2]

        results = self.yolo(
            frame,
            imgsz=self.img_size,
            verbose=False
        )[0]

        best_weapon = None

        best_assault_p = 0.0

        weapon_trigger = False

        assault_frame = False

        for box in results.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if (
                cls == self.person_class
                and conf >= self.person_conf
            ):

                roi = frame[
                    max(0, y1):min(h, y2),
                    max(0, x1):min(w, x2)
                ]

                if roi.size == 0:
                    continue

                assault_p, _ = self.gesture_predict(roi)

                if assault_p > best_assault_p:
                    best_assault_p = assault_p

                if assault_p >= self.assault_thresh:
                    assault_frame = True

                color = (
                    (0, 0, 255)
                    if assault_p >= self.assault_thresh
                    else (0, 255, 0)
                )

                label = (
                    "SUSPICIOUS"
                    if assault_p >= self.assault_thresh
                    else "NORMAL"
                )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    display,
                    f"{label} {assault_p:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            elif (
                cls in self.weapon_classes
                and conf >= self.weapon_conf
            ):

                weapon_trigger = True

                name = self.weapon_classes[cls]

                if (
                    best_weapon is None
                    or conf > best_weapon[1]
                ):

                    best_weapon = (
                        name,
                        conf
                    )

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"{name} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        self.assault_streak = (
            self.assault_streak + 1
            if assault_frame
            else 0
        )

        assault_trigger = (
            self.assault_streak >= self.assault_consec
        )

        if weapon_trigger:
            status = "DANGER"

        elif assault_trigger:
            status = "ALERT"

        else:
            status = "NORMAL"

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

        self.latency = (
            time.time() - t0
        ) * 1000

        self.latest_status = {
            "status": status,
            "fps": round(self.fps, 2),
            "latency_ms": round(self.latency, 2),
            "assault_streak": self.assault_streak
        }

        alert = (
            status != "NORMAL"
        )

        now = time.time()

        if (
            alert
            and (
                now - self.last_event_time
            ) >= self.cooldown_sec
        ):

            self.last_event_time = now

            eid = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            frame_path = os.path.join(
                self.out_frames,
                f"{eid}.jpg"
            )

            json_path = os.path.join(
                self.out_events,
                f"{eid}.json"
            )

            cv2.imwrite(
                frame_path,
                display
            )

            ts = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

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

            self.logs.appendleft(log_item)

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

        _, jpg = cv2.imencode(
            ".jpg",
            display
        )

        self.latest_frame = jpg.tobytes()