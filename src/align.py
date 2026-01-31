"""
Alignment demo using WORKING pipeline:
- Haar face detection (fast rough box)
- MediaPipe FaceMesh → 5 keypoints
- 5pt similarity transform → 112×112 aligned face
Run: python src/align.py
Keys: q → quit, s → save current aligned face
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

# ── 5-point indices from MediaPipe FaceMesh ──
LEFT_EYE    = 33
RIGHT_EYE   = 263
NOSE_TIP    = 1
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# ── Standard 112×112 target positions (ArcFace common reference) ──
# These are approximate; adjust if your book gives different ones
TARGET_POINTS = np.array([
    [38.2946,  51.6963],   # left eye
    [73.5318,  51.5014],   # right eye
    [56.0252,  71.7366],   # nose tip
    [41.5493,  92.3655],   # left mouth
    [70.7299,  92.2041]    # right mouth
], dtype=np.float32)


def align_face_5pt(image: np.ndarray, src_points: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute similarity transform (scale + rotation + translation) from 5 src points
    to standard target points, then warp the image.
    Returns: aligned image (or None), transformation matrix (or None)
    """
    if src_points.shape != (5, 2):
        return None, None

    # Enforce left/right ordering (safety)
    if src_points[0, 0] > src_points[1, 0]:
        src_points[[0, 1]] = src_points[[1, 0]]
    if src_points[3, 0] > src_points[4, 0]:
        src_points[[3, 4]] = src_points[[4, 3]]

    # Estimate similarity transform (least squares)
    M, _ = cv2.estimateAffinePartial2D(src_points, TARGET_POINTS[:5], method=cv2.RANSAC)

    if M is None:
        return None, None

    # Warp
    aligned = cv2.warpAffine(image, M, out_size, flags=cv2.INTER_LINEAR)
    return aligned, M


def put_text(img: np.ndarray, text: str, xy=(10, 30), scale=0.8, thickness=2, color=(255, 255, 255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(
    cam_index: int = 0,
    out_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Camera failed. Try index 1 or 2.")
        return

    # MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Haar for rough box (optional visualization)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)

    last_aligned = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    fps_t0 = time.time()
    fps_n = 0

    print("Alignment demo running. q = quit, s = save aligned crop")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        vis = frame.copy()
        H, W = frame.shape[:2]

        # ── Get 5 points via MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        aligned = None
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark
            idxs = [LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT]

            src_pts = np.array([[lm[i].x * W, lm[i].y * H] for i in idxs], dtype=np.float32)

            aligned, _ = align_face_5pt(frame, src_pts, out_size)

            if aligned is not None:
                last_aligned = aligned

        # ── Optional: draw Haar box for visualization ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haars = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))
        for (x, y, w, h) in haars:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # ── Status text ──
        if face_detected:
            put_text(vis, "Face OK → aligned", (10, 30), 0.75)
        else:
            put_text(vis, "No face", (10, 30), 0.9, color=(0, 0, 255))

        # FPS
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()
        put_text(vis, f"FPS: {fps:.1f}", (10, 60), 0.75)

        cv2.imshow("Camera + Detection", vis)
        cv2.imshow("Aligned 112x112", last_aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and last_aligned.size > 0:
            ts = int(time.time() * 1000)
            path = save_dir / f"aligned_{ts}.jpg"
            cv2.imwrite(str(path), last_aligned)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()