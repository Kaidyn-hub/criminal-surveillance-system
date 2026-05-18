# Handles file and directory paths
import os

# Used for FPS timing and delay
import time

# Flask modules for web application
from flask import (

    # Main Flask framework
    Flask,

    # Used for video streaming
    Response,

    # Used to render HTML pages
    render_template,

    # Used to send JSON data
    jsonify
)

# Import the main AI vision pipeline
from vision_pipeline import VisionPipeline

# INITIALIZE FLASK APPLICATION
app = Flask(__name__)

# GET CURRENT DIRECTORY
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# GET PROJECT ROOT DIRECTORY
PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

# YOLO MODEL PATH
YOLO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo",
    "yolo.pt"
)

# CNN MODEL PATH
CNN_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn",
    "cnn.pth"
)

# INITIALIZE AI PIPELINE
pipeline = VisionPipeline(

    # Load YOLO model
    yolo_weights=YOLO_PATH,

    # Load CNN model
    gesture_ckpt=CNN_PATH,

    # Use default webcam
    video_source=0,

    # Weapon detection confidence threshold
    weapon_conf=0.25,

    # Person detection confidence threshold
    person_conf=0.5,

    # Suspicious gesture threshold
    assault_thresh=0.5,

    # Required suspicious frames before alert
    assault_consec=0.5
)


# HOME PAGE ROUTE
@app.route("/")
def index():

    # Load index.html
    return render_template("index.html")

# VIDEO STREAM ROUTE
@app.route("/stream")
def stream():

    # FRAME GENERATOR FUNCTION
    def gen():

        # Limit FPS to improve performance
        FPS_LIMIT = 12

        # Calculate frame delay
        FRAME_DELAY = 1.0 / FPS_LIMIT

        # Infinite streaming loop
        while True:

            # Start timer
            t_start = time.time()

            # Run AI detection pipeline
            pipeline.step()

            # Skip if frame is empty
            if pipeline.latest_frame is None:

                time.sleep(0.01)

                continue

            # SEND VIDEO FRAME TO BROWSER
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + pipeline.latest_frame +
                b"\r\n"
            )

            # FPS CONTROL
            elapsed = (
                time.time() - t_start
            )

            # Add delay if processing is too fast
            if elapsed < FRAME_DELAY:

                time.sleep(
                    FRAME_DELAY - elapsed
                )

    # RETURN VIDEO STREAM RESPONSE
    return Response(

        # Frame generator
        gen(),

        # Streaming content type
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# STATUS API ROUTE
@app.route("/api/status")
def api_status():

    # Return latest system status
    return jsonify(
        pipeline.latest_status
    )

# DETECTION LOGS API ROUTE
@app.route("/api/logs")
def api_logs():

    # Return detection logs
    return jsonify(
        list(pipeline.logs)
    )

# SYSTEM HEALTH API ROUTE
@app.route("/api/health")
def api_health():

    # Return server and camera status
    return jsonify({

        # Server status
        "server": "running",

        # Camera connection status
        "camera": (
            "connected"
            if pipeline.cap.isOpened()
            else "not_connected"
        ),

        # Device used (CPU or GPU)
        "device": pipeline.device
    })


# RUN FLASK SERVER
if __name__ == "__main__":

    app.run(

        # Allow external access
        host="0.0.0.0",

        # Server port
        port=5000,

        # Disable debug mode
        debug=False,

        # Enable multithreading
        threaded=True
    )