import os
import time

from flask import (
    Flask,
    Response,
    render_template,
    jsonify
)

from vision_pipeline import VisionPipeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

YOLO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo",
    "yolo.pt"
)

CNN_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn",
    "cnn.pth"
)

pipeline = VisionPipeline(
    yolo_weights=YOLO_PATH,
    gesture_ckpt=CNN_PATH,
    video_source=0,
    weapon_conf=0.25,
    person_conf=0.5,
    assault_thresh=0.8
)

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/stream")
def stream():

    def gen():

        FPS_LIMIT = 12

        FRAME_DELAY = 1.0 / FPS_LIMIT

        while True:

            t_start = time.time()

            pipeline.step()

            if pipeline.latest_frame is None:

                time.sleep(0.01)

                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + pipeline.latest_frame +
                b"\r\n"
            )

            elapsed = (
                time.time() - t_start
            )

            if elapsed < FRAME_DELAY:

                time.sleep(
                    FRAME_DELAY - elapsed
                )

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/status")
def api_status():

    return jsonify(
        pipeline.latest_status
    )

@app.route("/api/logs")
def api_logs():

    return jsonify(
        list(pipeline.logs)
    )

@app.route("/api/health")
def api_health():

    return jsonify({
        "server": "running",
        "camera": (
            "connected"
            if pipeline.cap.isOpened()
            else "not_connected"
        ),
        "device": pipeline.device
    })

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )