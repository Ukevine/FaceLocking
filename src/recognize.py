"""
recognize.py - Live multi-face recognition demo
Haar multi-face → per-face MediaPipe 5pt → align → ArcFace ONNX → cosine match to DB
Run: python src/recognize.py
Keys:
  q     quit
  r     reload DB
  + / = increase distance threshold (looser matching)
  -     decrease distance threshold (stricter)
  d     toggle debug text
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

# ── Standard ArcFace 5-point target positions (112×112) ──
TARGET_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041]   # right mouth
], dtype=np.float32)


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # (5,2) float32


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# ── Alignment (same as align.py) ──
def align_face_5pt(image: np.ndarray, src_kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)):
    if src_kps.shape != (5, 2):
        return None, None

    if src_kps[0, 0] > src_kps[1, 0]:
        src_kps[[0, 1]] = src_kps[[1, 0]]
    if src_kps[3, 0] > src_kps[4, 0]:
        src_kps[[3, 4]] = src_kps[[4, 3]]

    M, _ = cv2.estimateAffinePartial2D(src_kps, TARGET_POINTS, method=cv2.RANSAC)
    if M is None:
        return None, None

    aligned = cv2.warpAffine(image, M, out_size, flags=cv2.INTER_LINEAR)
    return aligned, M


# ── Embedder (from embed.py) ──
class ArcFaceEmbedderONNX:
    def __init__(self, model_path="models/embedder_arcface.onnx", input_size=(112, 112)):
        self.in_w, self.in_h = input_size
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[:2] != (self.in_h, self.in_w):
            img_bgr = cv2.resize(img_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[np.newaxis, :].astype(np.float32)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.flatten().astype(np.float32)
        norm = np.linalg.norm(v) + 1e-12
        return (v / norm).astype(np.float32)


# ── Detection ──
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.idxs = [33, 263, 1, 61, 291]  # left eye, right eye, nose, mouth L/R

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haars = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))
        if len(haars) == 0:
            return []

        # Sort by size descending
        areas = haars[:, 2] * haars[:, 3]
        order = np.argsort(areas)[::-1]
        haars = haars[order][:max_faces]

        results = []
        H, W = frame.shape[:2]

        for (x, y, w, h) in haars:
            # Expand ROI slightly for better landmarks
            mx, my = int(0.25 * w), int(0.35 * h)
            rx1 = max(0, x - mx)
            ry1 = max(0, y - my)
            rx2 = min(W, x + w + mx)
            ry2 = min(H, y + h + my)
            roi = frame[ry1:ry2, rx1:rx2]

            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            kps_roi = np.array([[lm[i].x * (rx2-rx1), lm[i].y * (ry2-ry1)] for i in self.idxs], dtype=np.float32)
            kps = kps_roi + np.array([rx1, ry1])  # back to full frame

            # Simple validity check
            eye_dist = np.linalg.norm(kps[1] - kps[0])
            if eye_dist < 20:
                continue

            # Rough bbox from keypoints
            x1, y1 = np.min(kps, axis=0).astype(int) - 20
            x2, y2 = np.max(kps, axis=0).astype(int) + 20
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            results.append(FaceDet(x1, y1, x2, y2, kps))

        return results


# ── DB Matcher ──
class FaceMatcher:
    def __init__(self, db_path: Path, dist_threshold: float = 0.40):
        self.db_path = db_path
        self.dist_threshold = dist_threshold
        self.db: Dict[str, np.ndarray] = {}
        self.names: List[str] = []
        self.emb_matrix: Optional[np.ndarray] = None
        self._load_db()

    def _load_db(self):
        if not self.db_path.exists():
            print("No DB found. Enroll people first.")
            return
        data = np.load(self.db_path, allow_pickle=True)
        self.db = {k: data[k].astype(np.float32).ravel() for k in data.files}
        self.names = sorted(self.db.keys())
        if self.names:
            self.emb_matrix = np.stack([self.db[n] for n in self.names])

    def reload(self):
        self._load_db()
        print(f"DB reloaded: {len(self.names)} identities")

    def match(self, emb: np.ndarray) -> MatchResult:
        if self.emb_matrix is None or len(self.names) == 0:
            return MatchResult(None, 1.0, 0.0, False)

        sims = np.dot(self.emb_matrix, emb)
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])
        best_dist = 1.0 - best_sim
        accepted = best_dist <= self.dist_threshold
        name = self.names[best_idx] if accepted else None

        return MatchResult(name, best_dist, best_sim, accepted)


def main():
    db_path = Path("data/db/face_db.npz")
    if not db_path.exists():
        print("Database not found. Run enroll.py first.")
        return

    detector = FaceDetector()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceMatcher(db_path, dist_threshold=0.40)  # ← CHANGE THIS to your evaluated value!

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error.")
        return

    print("Live recognition running. q=quit, r=reload DB, +/- threshold, d=debug")

    fps_start = time.time()
    frame_count = 0
    show_debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        H, W = frame.shape[:2]

        faces = detector.detect(frame)

        for face in faces:
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            for pt in face.kps.astype(int):
                cv2.circle(vis, tuple(pt), 3, (0, 255, 0), -1)

            aligned, _ = align_face_5pt(frame, face.kps)
            if aligned is None:
                continue

            emb = embedder.embed(aligned)
            result = matcher.match(emb)

            label = result.name if result.accepted else "Unknown"
            color = (0, 255, 0) if result.accepted else (0, 0, 255)

            cv2.putText(vis, label, (face.x1, face.y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, f"dist {result.distance:.3f}", (face.x1, face.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # FPS
        frame_count += 1
        dt = time.time() - fps_start
        if dt > 1.0:
            fps = frame_count / dt
            frame_count = 0
            fps_start = time.time()
            cv2.putText(vis, f"FPS: {fps:.1f} | thr: {matcher.dist_threshold:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload()
        elif key in (ord("+"), ord("=")):
            matcher.dist_threshold = min(0.80, matcher.dist_threshold + 0.005)
            print(f"Threshold → {matcher.dist_threshold:.3f}")
        elif key == ord("-"):
            matcher.dist_threshold = max(0.10, matcher.dist_threshold - 0.005)
            print(f"Threshold → {matcher.dist_threshold:.3f}")
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()