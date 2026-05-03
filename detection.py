# ─── detection.py ───
# Per-frame detection: runs pose + tennis models and returns annotated frame.

from config import CONF_THRESHOLD, TENNIS_TARGET_CLASSES, COLORS
from drawing import draw_box, draw_pose_keypoints, draw_debug_overlay


def detect_frame(frame, pose_model, tennis_model, swing_detector):
    """
    Run all models on a single frame and draw bounding boxes + swing detection.

    IMPORTANT: All detections (person bbox, pose skeleton, swing) track the SAME person [0].

    Args:
        frame:          Raw BGR frame from OpenCV.
        pose_model:     Loaded YOLOv8 pose model.
        tennis_model:   Loaded CourtSide YOLO model.
        swing_detector: SwingDetector instance (stateful).

    Returns:
        (annotated_frame, swing_detected: bool)
    """
    # ── Pose detection (determines which person we track) ──
    pose_results = pose_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    swing_detected = False
    primary_person_bbox = None
    debug_info = None

    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
        keypoints_data = pose_results.keypoints[0]  # Primary player only

        if keypoints_data.xy is not None and keypoints_data.conf is not None:
            keypoints    = keypoints_data.xy.cpu().numpy()[0]    # (17, 2)
            confidences  = keypoints_data.conf.cpu().numpy()[0]  # (17,)

            draw_pose_keypoints(frame, keypoints, confidences)
            swing_detected, debug_info = swing_detector.detect_forehand_swing(keypoints, confidences)

            # debug_info passed back to main for separate debug window
            draw_debug_overlay(frame, debug_info)

            # Grab bbox for this same person from pose results
            if pose_results.boxes is not None and len(pose_results.boxes) > 0:
                primary_person_bbox = pose_results.boxes[0].xyxy[0]

    # ── Draw person bounding box (primary person only) ──
    if primary_person_bbox is not None:
        draw_box(frame, primary_person_bbox, "person (tracked)", COLORS["person"])

    # ── Tennis object detection (CourtSide) ──
    tennis_results = tennis_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    for box in tennis_results.boxes:
        cls = int(box.cls[0])
        if cls in TENNIS_TARGET_CLASSES:
            conf = float(box.conf[0])
            name = TENNIS_TARGET_CLASSES[cls]
            draw_box(frame, box.xyxy[0], f"{name} {conf:.0%}", COLORS[name])

    return frame, swing_detected, primary_person_bbox, debug_info
