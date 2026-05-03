# ─── swing_detector.py ───
# Detects forehand swings using elbow angle + two-phase timing logic.
#
# Phase 1 (Preparation) — tracked continuously every frame:
#   • Right wrist is on the RIGHT side of body center
#   • Elbow angle (wrist→elbow→shoulder) > ARM_EXTENDED_ANGLE (150°)
#   → Records last frame when each condition was true
#
# Phase 2 (Contact / Follow-through) — triggers a forehand count when:
#   • Right wrist Y is ABOVE left shoulder Y  (wrist_y < left_shoulder_y)
#   • Elbow angle < ARM_CONTACT_ANGLE (110°)
#   • last_frame_wrist_on_right  was within PREP_VALIDITY_SECONDS ago
#   • last_frame_arm_extended    was within PREP_VALIDITY_SECONDS ago
#   • Cooldown has passed

import numpy as np
from collections import deque
from config import (
    SMOOTHING_WINDOW, SWING_COOLDOWN, KEYPOINT_CONF_THRESHOLD,
    VIDEO_FPS, ARM_EXTENDED_ANGLE, ARM_CONTACT_ANGLE, PREP_VALIDITY_SECONDS,
)

# Pre-compute frame window from seconds
PREP_VALIDITY_FRAMES = int(PREP_VALIDITY_SECONDS * VIDEO_FPS)  # 30 frames


def _calc_angle(a, b, c):
    """
    Calculate the angle at point B formed by vectors BA and BC.
    a, b, c are (x, y) numpy arrays.
    Returns angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class SwingDetector:
    """Detects forehand swings using elbow angle and two-phase timing."""

    def __init__(self, smoothing_window=SMOOTHING_WINDOW, cooldown=SWING_COOLDOWN):
        self.smoothing_window = smoothing_window
        self.cooldown = cooldown

        # Keypoint history for temporal smoothing
        self.keypoint_history = {
            "right_shoulder": deque(maxlen=smoothing_window),
            "right_elbow":    deque(maxlen=smoothing_window),
            "right_wrist":    deque(maxlen=smoothing_window),
            "left_shoulder":  deque(maxlen=smoothing_window),
        }

        # Frame counters
        self.frame_number = 0
        self.frames_since_swing = cooldown  # Start ready to detect

        # Phase 1 timestamps (frame number of last occurrence)
        self.last_frame_wrist_on_right = -(PREP_VALIDITY_FRAMES + 1)
        self.last_frame_arm_extended   = -(PREP_VALIDITY_FRAMES + 1)

        # Result
        self.forehand_count = 0

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def detect_forehand_swing(self, keypoints, confidences):
        """
        Process one frame of keypoints.
        Returns: swing_detected (bool), debug_info (dict)
        """
        self.frame_number       += 1
        self.frames_since_swing += 1

        # ── Extract & smooth all needed keypoints ──
        right_shoulder = self._get_smoothed(keypoints, confidences, 6, "right_shoulder")
        right_elbow    = self._get_smoothed(keypoints, confidences, 8, "right_elbow")
        right_wrist    = self._get_smoothed(keypoints, confidences, 10, "right_wrist")
        left_shoulder  = self._get_smoothed(keypoints, confidences, 5, "left_shoulder")

        def fmt(pt):
            return (round(float(pt[0]), 1), round(float(pt[1]), 1)) if pt is not None else None

        # ── Default debug_info ──
        debug_info = {
            "right_wrist":    fmt(right_wrist),
            "right_shoulder": fmt(right_shoulder),
            "left_shoulder":  fmt(left_shoulder),
            "body_center_x":    None,
            "crossing_threshold": None,
            "elbow_angle":              None,
            "wrist_on_right_side":      None,
            "arm_extended":             None,
            "arm_at_contact":           None,
            "wrist_crossed_center":   None,
            "frames_since_wrist_right": None,
            "frames_since_arm_extended": None,
            "prep_valid":  None,
            "swing_phase": "waiting for keypoints",
        }

        # Need all four keypoints
        if any(kp is None for kp in [right_shoulder, right_elbow, right_wrist, left_shoulder]):
            return False, debug_info

        # ── Derived geometry ──
        body_center_x      = (right_shoulder[0] + left_shoulder[0]) / 2.0
        crossing_threshold = right_shoulder[0] + (1/2) * abs(body_center_x - right_shoulder[0])
        debug_info["body_center_x"]      = round(float(body_center_x), 1)
        debug_info["crossing_threshold"] = round(float(crossing_threshold), 1)

        elbow_angle = _calc_angle(
            np.array(right_wrist),
            np.array(right_elbow),
            np.array(right_shoulder),
        )
        debug_info["elbow_angle"] = round(elbow_angle, 1)
        if left_shoulder[0] > right_shoulder[0]:
            # Person facing LEFT in frame
            wrist_on_right_side  = right_wrist[0] < right_shoulder[0]
            wrist_crossed_center = right_wrist[0] > crossing_threshold
        else:
            # Person facing RIGHT in frame
            wrist_on_right_side  = right_wrist[0] > right_shoulder[0]
            wrist_crossed_center = right_wrist[0] < crossing_threshold
        arm_extended   = elbow_angle > ARM_EXTENDED_ANGLE   # Phase 1b: arm stretched
        arm_at_contact = elbow_angle < ARM_CONTACT_ANGLE    # Phase 2b: arm bent at contact

        debug_info["wrist_on_right_side"]    = wrist_on_right_side
        debug_info["arm_extended"]           = arm_extended
        debug_info["arm_at_contact"]         = arm_at_contact
        debug_info["wrist_crossed_center"] = wrist_crossed_center

        # ── Update Phase 1 timestamps ──
        if wrist_on_right_side:
            self.last_frame_wrist_on_right = self.frame_number
        if arm_extended:
            self.last_frame_arm_extended = self.frame_number

        frames_since_wrist_right  = self.frame_number - self.last_frame_wrist_on_right
        frames_since_arm_extended = self.frame_number - self.last_frame_arm_extended

        debug_info["frames_since_wrist_right"]   = frames_since_wrist_right
        debug_info["frames_since_arm_extended"]  = frames_since_arm_extended

        prep_valid = (
            frames_since_wrist_right  <= PREP_VALIDITY_FRAMES and
            frames_since_arm_extended <= PREP_VALIDITY_FRAMES
        )
        debug_info["prep_valid"] = prep_valid

        # ── Swing phase label ──
        if wrist_on_right_side and arm_extended:
            debug_info["swing_phase"] = "prep: arm extended right"
        elif wrist_on_right_side:
            debug_info["swing_phase"] = "prep: wrist right (arm not extended)"
        elif prep_valid:
            debug_info["swing_phase"] = "transitioning..."
        else:
            debug_info["swing_phase"] = "idle"

        # ── Phase 2: Forehand trigger ──
        if (wrist_crossed_center and
                arm_at_contact and
                prep_valid and
                self.frames_since_swing >= self.cooldown):

            self.forehand_count    += 1
            self.frames_since_swing = 0
            debug_info["swing_phase"] = "FOREHAND DETECTED!"
            return True, debug_info

        return False, debug_info

    def get_count(self):
        return self.forehand_count

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def smooth_keypoint(self, name, position):
        """Append position to history and return smoothed mean."""
        if position is not None:
            self.keypoint_history[name].append(position)
        if len(self.keypoint_history[name]) > 0:
            return np.mean(np.array(self.keypoint_history[name]), axis=0)
        return None

    def _get_smoothed(self, keypoints, confidences, idx, name):
        """Return smoothed keypoint if confident, else last known value."""
        if confidences[idx] >= KEYPOINT_CONF_THRESHOLD:
            return self.smooth_keypoint(name, keypoints[idx])
        return self.smooth_keypoint(name, None)
