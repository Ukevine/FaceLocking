"""
Haar face detection + practical 5-point landmarks (MediaPipe FaceMesh).
Why this works:
- Haar: fast CPU detection
- FaceMesh: confirms real face + stable 5 points
- Rejects false positives
- Centered bbox from keypoints
- EMA smoothing for stability
Run: python src/haar_5pt.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# ── Data ──
@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32

# ── Alignment Helpers ──
def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Similarity transform: maps detected 5 points → ArcFace template.
    Falls back to 3 points (eyes + nose) if 5pt fails.
    """
    k = kps_5x2.astype(np.float32)
    
    # ArcFace standard template (112×112)
    dst = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041]    # right mouth
    ], dtype=np.float32)

    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        dst *= np.array([out_w / 112.0, out_h / 112.0], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        # Fallback: use eyes + nose only
        M = cv2.getAffineTransform(
            k[[0, 1, 2]],
            dst[[0, 1, 2]]
        )
    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Warp frame to aligned view using 5-point transform."""
    if kps_5x2.shape != (5, 2):
        return None, None

    M = _estimate_norm_5pt(kps_5x2, out_size)
    if M is None:
        return None, None

    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return aligned, M


# ── Utility ──
def _clip_box_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = b.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15
) -> np.ndarray:
    """Centered bbox with more forehead/chin padding."""
    k = kps.astype(np.float32)
    x_min, y_min = np.min(k, axis=0)
    x_max, y_max = np.max(k, axis=0)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """Basic geometry filter."""
    le, re, no, lm, rm = kps.astype(np.float32)
    eye_dist = np.linalg.norm(re - le)
    if eye_dist < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# ── Detector Class ──
class Haar5ptDetector:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = False,
    ):
        self.debug = debug
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = smooth_alpha

        # Haar
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        # MediaPipe
        if mp is None:
            raise RuntimeError(f"MediaPipe import failed: {_MP_IMPORT_ERROR}\nInstall: pip install mediapipe")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_LEFT_EYE    = 33
        self.IDX_RIGHT_EYE   = 263
        self.IDX_NOSE_TIP    = 1
        self.IDX_MOUTH_LEFT  = 61
        self.IDX_MOUTH_RIGHT = 291

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_size
        )
        if len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP,
                self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]

        pts = [[lm[i].x * frame_bgr.shape[1], lm[i].y * frame_bgr.shape[0]] for i in idxs]
        kps = np.array(pts, dtype=np.float32)

        # Enforce left/right
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)

        if len(faces) == 0:
            return []

        # Take largest face only (for simplicity in this module)
        areas = faces[:, 2] * faces[:, 3]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]

        kps = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[haar_5pt] Haar found, but FaceMesh failed → reject")
            return []

        if not _kps_span_ok(kps, min_eye_dist=max(12.0, 0.18 * w)):
            if self.debug:
                print("[haar_5pt] 5pt geometry invalid → reject")
            return []

        box = _bbox_from_5pt(kps)
        box = _clip_box_xyxy(box, W, H)

        # EMA smoothing
        box_s = _ema(self._prev_box, box, self.smooth_alpha)
        kps_s = _ema(self._prev_kps, kps, self.smooth_alpha)
        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        return [FaceKpsBox(
            x1=int(round(box_s[0])),
            y1=int(round(box_s[1])),
            x2=int(round(box_s[2])),
            y2=int(round(box_s[3])),
            score=1.0,
            kps=kps_s
        )]


# ── Demo ──
def main():
    if mp is None:
        print("MediaPipe not installed. Run: pip install mediapipe")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return

    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True
    )

    print("Haar + FaceMesh 5pt demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = det.detect(frame, max_faces=1)
        vis = frame.copy()

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for pt in f.kps.astype(int):
                cv2.circle(vis, tuple(pt), 4, (0, 255, 0), -1)
            cv2.putText(vis, "Face OK", (f.x1, f.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No face", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Haar + 5pt Landmarks", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()