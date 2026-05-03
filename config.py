# ─── config.py ───
# All constants and configuration for the Tennis Detector

# ─── Target classes ───
# From CourtSide (tennis_model): class 0 = racket, class 1 = tennis_ball
TENNIS_TARGET_CLASSES = {0: "racket"}

# ─── Pose Keypoint Indices (COCO format) ───
# 0: nose, 1-2: eyes, 3-4: ears
# 5: left shoulder, 6: right shoulder
# 7: left elbow,    8: right elbow
# 9: left wrist,   10: right wrist
# 11: left hip,    12: right hip
# 13: left knee,   14: right knee
# 15: left ankle,  16: right ankle
KEYPOINT_NAMES = {
    5: "left_shoulder",  6: "right_shoulder",
    7: "left_elbow",     8: "right_elbow",
    9: "left_wrist",    10: "right_wrist",
}

# ─── Colors (BGR) ───
COLORS = {
    "person": (0, 255, 0),   # Green
    "racket": (0, 0, 255),   # Red
}

# ─── Detection thresholds ───
CONF_THRESHOLD = 0.25
KEYPOINT_CONF_THRESHOLD = 0.5  # Confidence threshold for keypoints

# ─── Swing detection parameters ───
SMOOTHING_WINDOW = 4     # Frames used for temporal smoothing
SWING_COOLDOWN = 60      # Frames to wait before detecting next swing (60fps → 1s cooldown)
VIDEO_FPS = 60           # Expected video FPS — used for timing windows

# Phase 1 — Preparation conditions (must have occurred within PREP_VALIDITY_SECONDS)
ARM_EXTENDED_ANGLE = 135   # Elbow angle (degrees) must be ABOVE this → arm stretched
# Wrist must also be on right side of body center during prep

# Phase 2 — Contact/follow-through trigger
ARM_CONTACT_ANGLE = 125    # Elbow angle (degrees) must be BELOW this at contact
# Wrist must be above left shoulder Y (wrist_y < left_shoulder_y)

# How long Phase 1 conditions remain "valid" before Phase 2 must fire
PREP_VALIDITY_SECONDS = 0.5   # 0.5s × 60fps = 30 frames
