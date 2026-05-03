# ─── pose_capturer.py ───
# Captures and beautifies sport-quality photos on forehand detection.
#
# Pipeline per saved photo:
#   1. Sample 3 random frames from a window of N-15 to N+5 around detection
#   2. For each frame: expand bbox with padding → normalize to 3:2 aspect ratio
#   3. Beautify: denoise → CLAHE → saturation boost → unsharp mask (sharpen)
#   4. Save as high-quality JPG

import cv2
import numpy as np
import os
import random
from collections import deque
from datetime import datetime

# ─── Capturer config ───
FRAME_BUFFER_SIZE   = 20       # Must be >= abs(WINDOW_START), i.e. 15
WINDOW_START        = -15      # Frames before detection
WINDOW_END          = 5        # Frames after detection (needs post-buffer)
SAMPLES_PER_SWING   = 3        # Random frames to save per forehand

BBOX_PADDING        = 0.45     # Expand bbox by this fraction on each side
ASPECT_RATIO        = 3 / 2    # Target crop ratio (width / height) — DSLR sport standard

# Beautify params
JPG_QUALITY         = 97       # JPEG save quality (0-100)

# S-curve tone mapping (0-255 input → output, controls contrast feel)
# Crushes blacks slightly, lifts mids, protects highlights
TONE_CURVE_IN   = [0,  30,  80, 128, 200, 245, 255]
TONE_CURVE_OUT  = [0,  15,  75, 138, 210, 250, 255]

# Vibrance (protects skin tones unlike raw saturation)
VIBRANCE_STRENGTH   = 0.45     # 0 = no change, 1 = full boost

# Edge-aware sharpening
SHARPEN_SIGMA       = 0.8      # Gaussian blur sigma for unsharp mask
SHARPEN_AMOUNT      = 1.4      # Sharpening strength
SHARPEN_EDGE_THRESH = 10       # Only sharpen pixels with luminance gradient above this

# Color grading — sport Instagram look
# Warm highlights: push R up, B down in bright areas
# Cool shadows: push B up, R down in dark areas
GRADE_HIGHLIGHT_WARM = 0.06    # Fraction to shift highlights toward warm
GRADE_SHADOW_COOL    = 0.05    # Fraction to shift shadows toward cool/teal

# Vignette
VIGNETTE_STRENGTH   = 0.45     # 0 = none, 1 = very strong dark edges

OUTPUT_DIR          = "captures"  # Folder to save photos


class PoseCapturer:
    """
    Buffers raw frames and bboxes, then on forehand detection:
    samples random frames from the window, beautifies and saves them.
    """

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Rolling buffer of (raw_frame, bbox_or_None)
        # Stores last FRAME_BUFFER_SIZE frames (the pre-detection window)
        self.pre_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

        # Post-detection buffer: filled after a swing fires
        self.post_buffer      = []       # List of (frame, bbox)
        self.post_frames_needed = 0      # Countdown of how many post frames still needed
        self.pending_capture  = False    # True while collecting post frames

        self.swing_count = 0

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def push_frame(self, raw_frame, bbox):
        """
        Call every frame with the raw (un-annotated) frame and person bbox.
        bbox can be None if person not detected.
        """
        entry = (raw_frame.copy(), bbox)

        # Always feed pre-buffer
        self.pre_buffer.append(entry)

        # If we're collecting post-detection frames
        if self.pending_capture:
            self.post_buffer.append(entry)
            self.post_frames_needed -= 1

            if self.post_frames_needed <= 0:
                self._process_capture()
                self.pending_capture = False
                self.post_buffer     = []

    def on_swing_detected(self):
        """
        Call when SwingDetector fires. Starts collecting post frames.
        Pre frames are already in the buffer.
        """
        self.swing_count    += 1
        self.post_buffer     = []
        self.post_frames_needed = max(WINDOW_END, 0)
        self.pending_capture = True
        print(f"[CAPTURER] Forehand #{self.swing_count} detected — collecting {self.post_frames_needed} post frames...")

    # ──────────────────────────────────────────────────────────────
    # Internal: sample → crop → beautify → save
    # ──────────────────────────────────────────────────────────────

    def _process_capture(self):
        """Build full window, sample 3 random frames, beautify and save each."""
        # Full window = last FRAME_BUFFER_SIZE pre frames + post frames
        # pre_buffer already has up to FRAME_BUFFER_SIZE entries (rolling)
        full_window = list(self.pre_buffer) + self.post_buffer

        if len(full_window) == 0:
            print("[CAPTURER] Warning: empty window, skipping capture.")
            return

        # Sample up to SAMPLES_PER_SWING random frames
        n_samples = min(SAMPLES_PER_SWING, len(full_window))
        sampled   = random.sample(full_window, n_samples)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (frame, bbox) in enumerate(sampled):
            cropped    = _crop_frame(frame, bbox)
            beautified = _beautify(cropped)

            filename = os.path.join(
                self.output_dir,
                f"forehand_{self.swing_count:03d}_frame{i+1}_{timestamp}.jpg"
            )
            cv2.imwrite(filename, beautified, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
            print(f"[CAPTURER] Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────
# Crop helper
# ──────────────────────────────────────────────────────────────────────

def _crop_frame(frame, bbox):
    """
    Expand bbox by BBOX_PADDING, normalize to 3:2 aspect ratio,
    clamp to frame boundaries, and return the cropped region.
    Falls back to full frame if no bbox.
    """
    h, w = frame.shape[:2]

    if bbox is None:
        # No bbox — use full frame, just normalize aspect ratio
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = map(float, bbox)

    # ── Expand by padding ──
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * BBOX_PADDING
    pad_y = bh * BBOX_PADDING

    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y

    # ── Normalize to 3:2 aspect ratio ──
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w / crop_h > ASPECT_RATIO:
        # Too wide — extend height
        crop_h = crop_w / ASPECT_RATIO
    else:
        # Too tall — extend width
        crop_w = crop_h * ASPECT_RATIO

    x1 = cx - crop_w / 2
    x2 = cx + crop_w / 2
    y1 = cy - crop_h / 2
    y2 = cy + crop_h / 2

    # ── Clamp to frame ──
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    return frame[y1:y2, x1:x2]


# ──────────────────────────────────────────────────────────────────────
# Beautify pipeline
# ──────────────────────────────────────────────────────────────────────

def _build_tone_lut():
    """Build a 256-entry LUT from the tone curve control points."""
    lut = np.interp(np.arange(256), TONE_CURVE_IN, TONE_CURVE_OUT).astype(np.uint8)
    return lut

_TONE_LUT = _build_tone_lut()


def _beautify(frame):
    """
    Instagram sport-photography enhancement pipeline:
      1. S-curve tone mapping      — punchy contrast, cinematic feel
      2. Vibrance boost            — vivid colors, natural skin tones
      3. Color grading             — warm highlights + cool/teal shadows
      4. Edge-aware sharpening     — crisp edges, smooth skin/background
      5. Vignette                  — draws eye to the athlete
    """
    img = frame.astype(np.float32)

    # ── 1. S-curve tone mapping (per channel via LUT) ──
    toned = cv2.LUT(frame, _TONE_LUT)

    # ── 2. Vibrance — boost only muted colors, protect saturated ones ──
    hsv = cv2.cvtColor(toned, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    # Vibrance boost is inversely proportional to existing saturation
    boost = VIBRANCE_STRENGTH * (1.0 - s)
    hsv[:, :, 1] = np.clip((s + boost) * 255.0, 0, 255)
    vibrant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ── 3. Color grading — warm highlights, cool shadows ──
    # Luminance mask: 0=shadows, 1=highlights
    lum = cv2.cvtColor(vibrant.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    highlight_mask = lum[:, :, np.newaxis]       # bright areas
    shadow_mask    = (1.0 - lum)[:, :, np.newaxis]  # dark areas

    graded = vibrant.copy()
    # Warm highlights: push R up, B down
    graded[:, :, 2] += highlight_mask[:, :, 0] * GRADE_HIGHLIGHT_WARM * 255  # R up
    graded[:, :, 0] -= highlight_mask[:, :, 0] * GRADE_HIGHLIGHT_WARM * 128  # B down
    # Cool shadows: push B up, R down slightly
    graded[:, :, 0] += shadow_mask[:, :, 0] * GRADE_SHADOW_COOL * 255        # B up
    graded[:, :, 2] -= shadow_mask[:, :, 0] * GRADE_SHADOW_COOL * 128        # R down
    graded = np.clip(graded, 0, 255).astype(np.uint8)

    # ── 4. Edge-aware sharpening ──
    gray     = cv2.cvtColor(graded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred  = cv2.GaussianBlur(graded, (0, 0), SHARPEN_SIGMA)
    # Edge mask: only sharpen where gradient is strong
    grad_x   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = np.clip(edge_mag / (SHARPEN_EDGE_THRESH * 10), 0, 1)[:, :, np.newaxis]

    sharp_full = cv2.addWeighted(graded, 1 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)
    sharpened  = (graded.astype(np.float32) * (1 - edge_mask) +
                  sharp_full.astype(np.float32) * edge_mask)
    sharpened  = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── 5. Vignette — darken edges to focus on athlete ──
    h, w = sharpened.shape[:2]
    # Build elliptical gradient: 1 at center, 0 at corners
    cx, cy   = w / 2, h / 2
    Y, X     = np.ogrid[:h, :w]
    dist     = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - VIGNETTE_STRENGTH * (dist - 0.5), 0, 1)[:, :, np.newaxis]

    result = np.clip(sharpened.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    return result
