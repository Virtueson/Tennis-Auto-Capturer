# ─── drawing.py ───
# All OpenCV drawing helpers for the Tennis Detector.

import cv2
import numpy as np
from collections import deque
from config import KEYPOINT_CONF_THRESHOLD

# ── Scrolling graph history (last N frames) ──
_GRAPH_HISTORY = 120  # frames to show in scrolling graph
_graph_data = {
    "elbow_angle":              deque(maxlen=_GRAPH_HISTORY),
    "frames_since_wrist_right": deque(maxlen=_GRAPH_HISTORY),
    "frames_since_arm_extended":deque(maxlen=_GRAPH_HISTORY),
}
_GRAPH_COLORS = {
    "elbow_angle":               (0, 220, 220),   # Cyan
    "frames_since_wrist_right":  (0, 200, 0),     # Green
    "frames_since_arm_extended": (0, 100, 255),   # Orange
}
_GRAPH_RANGES = {
    "elbow_angle":               (0, 180),
    "frames_since_wrist_right":  (0, 60),
    "frames_since_arm_extended": (0, 60),
}


# ══════════════════════════════════════════════════════════════════════
# Video frame helpers (drawn ON the video frame)
# ══════════════════════════════════════════════════════════════════════

def draw_box(frame, box_xyxy, label, color):
    """Draw a bounding box and label on the frame."""
    x1, y1, x2, y2 = map(int, box_xyxy)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y1, label_size[1] + 5)
    cv2.rectangle(
        frame,
        (x1, label_y - label_size[1] - 5),
        (x1 + label_size[0] + 10, label_y + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )


def draw_pose_keypoints(frame, keypoints, confidences):
    """Draw skeleton keypoints and connections on the frame."""
    skeleton = [
        (5, 7), (7, 9),    # Left arm
        (6, 8), (8, 10),   # Right arm
        (5, 6),            # Shoulders
        (5, 11), (6, 12),  # Torso
        (11, 12),          # Hips
    ]
    for pt1_idx, pt2_idx in skeleton:
        if (confidences[pt1_idx] >= KEYPOINT_CONF_THRESHOLD and
                confidences[pt2_idx] >= KEYPOINT_CONF_THRESHOLD):
            pt1 = tuple(keypoints[pt1_idx].astype(int))
            pt2 = tuple(keypoints[pt2_idx].astype(int))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

    for kp, conf in zip(keypoints, confidences):
        if conf >= KEYPOINT_CONF_THRESHOLD:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)


def draw_counter(frame, count, swing_detected):
    """Draw forehand swing counter on the top-left corner of the frame."""
    cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 70), (255, 255, 255), 2)

    text = f"Forehand Swings: {count}"
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if swing_detected:
        cv2.circle(frame, (280, 40), 15, (0, 255, 0), -1)


def draw_debug_overlay(frame, debug_info):
    """
    Previously drew debug info on the video frame.
    Now a no-op — debug info is shown in the separate debug window via draw_debug_window().
    Kept so detection.py doesn't need changes.
    """
    pass


# ══════════════════════════════════════════════════════════════════════
# Separate debug window (drawn on its OWN canvas, shown via imshow)
# ══════════════════════════════════════════════════════════════════════

# Debug window canvas size
_DBG_W = 600
_DBG_H = 720

# Text panel rows — same info as the old overlay
_ROWS = [
    ("── Coordinates ──",        None,                         "header"),
    ("right_wrist",              "right_wrist",                "coord"),
    ("right_shoulder",           "right_shoulder",             "coord"),
    ("left_shoulder",            "left_shoulder",              "coord"),
    ("body_center_x",            "body_center_x",              "scalar"),
    ("crossing_threshold",       "crossing_threshold",         "scalar"),
    ("── Angle ──",              None,                         "header"),
    ("elbow_angle (deg)",        "elbow_angle",                "scalar"),
    ("── Phase 1 Conditions ──", None,                         "header"),
    ("wrist_on_right_side",      "wrist_on_right_side",        "bool"),
    ("arm_extended (>1535)",      "arm_extended",               "bool"),
    ("frm since wrist right",    "frames_since_wrist_right",   "scalar"),
    ("frm since arm extended",   "frames_since_arm_extended",  "scalar"),
    ("prep_valid (<= 30frm)",    "prep_valid",                 "bool"),
    ("── Phase 2 Trigger ──",    None,                         "header"),
    ("wrist_crossed_center",     "wrist_crossed_center",       "bool"),
    ("arm_at_contact (<125)",    "arm_at_contact",             "bool"),
    ("── Phase ──",              None,                         "header"),
    ("swing_phase",              "swing_phase",                "text"),
]

# Layout constants
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_LINE_H    = 26
_MARGIN    = 10
_LABEL_X   = 12
_VALUE_X   = 320
_TEXT_PANEL_H = _MARGIN + (_LINE_H * (len(_ROWS) + 1)) + _MARGIN  # auto height for text section


def draw_debug_window(debug_info):
    """
    Build and return a 600x720 debug canvas with:
      - Top: text panel showing all debug values (same info as old overlay)
      - Bottom: scrolling line graph for elbow_angle, frames_since_wrist/arm
    Call cv2.imshow('Debug', draw_debug_window(debug_info)) in main.py.
    """
    canvas = np.zeros((_DBG_H, _DBG_W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)  # dark background

    # ── Update graph history ──
    if debug_info is not None:
        for key in _graph_data:
            val = debug_info.get(key)
            _graph_data[key].append(float(val) if val is not None else 0.0)

    # ══ TEXT PANEL ══
    _draw_text_panel(canvas, debug_info)

    # ── Divider ──
    cv2.line(canvas, (0, _TEXT_PANEL_H), (_DBG_W, _TEXT_PANEL_H), (80, 80, 80), 1)

    # ══ GRAPH PANEL ══
    graph_top = _TEXT_PANEL_H + 5
    _draw_graph_panel(canvas, graph_top)

    return canvas


def _draw_text_panel(canvas, debug_info):
    """Draw all debug key-value rows onto canvas."""
    # Title
    cv2.putText(canvas, "SWING DEBUG", (_LABEL_X, _MARGIN + 18),
                _FONT, 0.65, (255, 220, 0), 1, cv2.LINE_AA)

    for i, (label, key, kind) in enumerate(_ROWS):
        y = _MARGIN + (i + 2) * _LINE_H

        if kind == "header":
            cv2.putText(canvas, label, (_LABEL_X, y),
                        _FONT, 0.40, (130, 130, 130), 1, cv2.LINE_AA)
            continue

        val = debug_info.get(key) if debug_info and key else None

        # Format value and color
        if val is None:
            val_str, val_color = "None", (100, 100, 100)
        elif kind == "coord":
            val_str   = f"({val[0]:.0f}, {val[1]:.0f})"
            val_color = (0, 220, 220)
        elif kind == "bool":
            val_str   = str(val)
            val_color = (0, 210, 0) if val else (0, 0, 210)
        elif kind == "scalar":
            val_str   = str(val)
            val_color = (0, 220, 220)
        else:  # text
            val_str   = str(val)
            val_color = (0, 255, 255) if "DETECTED" in val_str else (220, 220, 220)

        cv2.putText(canvas, f"{label}:", (_LABEL_X, y),
                    _FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, val_str, (_VALUE_X, y),
                    _FONT, 0.45, val_color, 1, cv2.LINE_AA)


def _draw_graph_panel(canvas, top):
    """Draw scrolling line graphs for elbow_angle and frame-since counters."""
    graph_h     = _DBG_H - top - 10   # remaining height
    graph_w     = _DBG_W - 60         # leave left margin for labels
    graph_x     = 50                  # left margin
    graph_y     = top

    # Background
    cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                  (35, 35, 35), -1)

    # Title
    cv2.putText(canvas, "Live Graph", (graph_x, graph_y - 4),
                _FONT, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

    # Draw each series
    for key, color in _GRAPH_COLORS.items():
        data  = list(_graph_data[key])
        if len(data) < 2:
            continue
        vmin, vmax = _GRAPH_RANGES[key]
        vrange = vmax - vmin if vmax != vmin else 1

        pts = []
        for j, v in enumerate(data):
            px = graph_x + int(j / _GRAPH_HISTORY * graph_w)
            py = graph_y + graph_h - int(((v - vmin) / vrange) * graph_h)
            py = max(graph_y, min(graph_y + graph_h, py))
            pts.append((px, py))

        for j in range(1, len(pts)):
            cv2.line(canvas, pts[j-1], pts[j], color, 1, cv2.LINE_AA)

        # Legend label (right side, stacked)
        legend_idx = list(_GRAPH_COLORS.keys()).index(key)
        lx = graph_x + graph_w + 4
        ly = graph_y + 14 + legend_idx * 18
        cv2.putText(canvas, key.replace("frames_since_", "frm_").replace("_", " "),
                    (4, ly), _FONT, 0.30, color, 1, cv2.LINE_AA)

    # Y-axis ticks for elbow_angle (0, 90, 180)
    for tick_val in [0, 90, 180]:
        ty = graph_y + graph_h - int((tick_val / 180) * graph_h)
        cv2.line(canvas, (graph_x - 4, ty), (graph_x, ty), (80, 80, 80), 1)
        cv2.putText(canvas, str(tick_val), (graph_x - 30, ty + 4),
                    _FONT, 0.30, (100, 100, 100), 1, cv2.LINE_AA)
