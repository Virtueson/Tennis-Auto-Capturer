# ─── main.py ───
# Entry point for the Tennis Detector.

import cv2
from ultralytics import YOLO
from huggingface_hub import snapshot_download

from swing_detector import SwingDetector
from detection import detect_frame
from drawing import draw_counter, draw_debug_window
from pose_capturer import PoseCapturer


# ─── Download CourtSide model from HuggingFace ───
snapshot_download(
    repo_id="Davidsv/CourtSide-Computer-Vision-v1",
    local_dir="./CourtSide-Computer-Vision-v1",
    local_dir_use_symlinks=False,
)

# ─── Load Models ───
tennis_model = YOLO("CourtSide-Computer-Vision-v1/model.pt")  # CourtSide → racket & ball
pose_model   = YOLO("yolov8n-pose.pt")                        # Pose model → keypoint detection


def main():
    # ── Video input path ── change this to your video file
    video_path = "video 3.mp4"   # <── PUT YOUR VIDEO PATH HERE

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS")
    print("[INFO] Press 'q' to quit")

    swing_detector = SwingDetector()
    pose_capturer  = PoseCapturer(output_dir="captures")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        frame, swing_detected, bbox, debug_info = detect_frame(raw_frame.copy(), pose_model, tennis_model, swing_detector)

        # Feed raw frame + bbox into capturer buffer every frame
        pose_capturer.push_frame(raw_frame, bbox)

        # Trigger capture on swing
        if swing_detected:
            pose_capturer.on_swing_detected()

        draw_counter(frame, swing_detector.get_count(), swing_detected)

        # ── Video window (resized for display) ──
        display_frame = cv2.resize(frame, (848, 478))
        cv2.imshow("Tennis Detector", display_frame)

        # ── Separate debug window ──
        debug_canvas = draw_debug_window(debug_info)
        cv2.imshow("Swing Debug", debug_canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUMMARY] Total Forehand Swings Detected: {swing_detector.get_count()}")


if __name__ == "__main__":
    main()
