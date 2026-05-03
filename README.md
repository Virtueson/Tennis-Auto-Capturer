# Tennis Forehand Swing Detector

Real-time tennis forehand swing counter and photo capturer. Processes a video file using two YOLO models — pose estimation and court/object detection — then applies a two-phase algorithm to count forehand swings and auto-save sport-quality cropped photos on each detection.

![Demo](demo.gif)

## Features

- Real-time forehand swing detection using body keypoint geometry
- Two-window display: annotated video + live debug graph (elbow angle, prep state)
- Auto-saves 3 beautified photos per swing (S-curve, vibrance, color grading, sharpening, vignette)
- Swing counter overlay on video

## How It Works

**Two models run on every frame:**
- `yolov8n-pose.pt` — YOLOv8 pose model, extracts 17 COCO skeleton keypoints for the primary player
- `CourtSide-Computer-Vision-v1` (auto-downloaded from HuggingFace) — YOLOv11n, detects racket, tennis ball, and 8 court zones

**Two-phase swing FSM:**
1. **Prep phase** — right wrist is on the racket side of the body AND elbow angle > 135°
2. **Contact phase** — wrist crosses the body center AND elbow angle < 125°, within 0.5s of Phase 1

## Setup

```bash
pip install ultralytics huggingface_hub opencv-python numpy
```

Download the YOLOv8 pose model:
```bash
# It will auto-download on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

The CourtSide detection model downloads automatically from HuggingFace on first run.

## Usage

1. Edit `main.py` and set `video_path` to your tennis video file:
   ```python
   video_path = "your_video.mp4"
   ```

2. Run:
   ```bash
   python main.py
   ```

3. Press `q` to quit. Swing photos are saved to `./captures/`.

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARM_EXTENDED_ANGLE` | 135° | Elbow angle threshold for prep phase |
| `ARM_CONTACT_ANGLE` | 125° | Elbow angle threshold for contact phase |
| `PREP_VALIDITY_SECONDS` | 0.5s | How long prep phase stays valid |
| `SWING_COOLDOWN` | 60 frames | Minimum frames between swings (1s @ 60fps) |
| `VIDEO_FPS` | 60 | Set this to match your video's FPS |
| `CONF_THRESHOLD` | 0.25 | YOLO detection confidence threshold |

## Project Structure

```
main.py             # Entry point — video loop, model loading
config.py           # All constants and thresholds
detection.py        # Per-frame model inference
swing_detector.py   # Two-phase forehand FSM
drawing.py          # OpenCV drawing helpers + debug window
pose_capturer.py    # Buffered frame capture + photo beautify pipeline
```

## Models

| Model | Source | Purpose |
|-------|--------|---------|
| `yolov8n-pose.pt` | Ultralytics (auto-download) | 17-point skeleton keypoints |
| CourtSide CV v1 | [HuggingFace: Davidsv/CourtSide-Computer-Vision-v1](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1) | Racket + ball + court zones (10 classes, mAP@50: 92.1%) |

## Requirements

- Python 3.8+
- `ultralytics`
- `huggingface_hub`
- `opencv-python`
- `numpy`
