"""
Embedding stage (ArcFace ONNX)
camera → Haar rough box → MediaPipe 5pt → align 112×112 → ArcFace embedding → viz
Run: python src/embed.py
Keys: q → quit, p → print embedding stats
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# ── MediaPipe 5-point indices ──
LEFT_EYE    = 33
RIGHT_EYE   = 263
NOSE_TIP    = 1
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# Standard target points for ArcFace-style alignment (112×112)
TARGET_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)


@dataclass
class EmbeddingResult:
    embedding: np.ndarray      # (512,) float32, L2-normalized
    norm_before: float
    dim: int


class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug
        
        # Load ONNX model (CPU only)
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        
        if debug:
            print("[ArcFace ONNX] Model loaded")
            print("  Input shape:", self.sess.get_inputs()[0].shape)
            print("  Output shape:", self.sess.get_outputs()[0].shape)

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[:2] != (self.in_h, self.in_w):
            img_bgr = cv2.resize(img_bgr, (self.in_w, self.in_h))
        
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0                     # ArcFace standard normalization
        x = np.transpose(rgb, (2, 0, 1))[np.newaxis, :] # (1, 3, 112, 112)
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32), n

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        x = self._preprocess(aligned_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.flatten().astype(np.float32)
        v_norm, norm_before = self._l2_normalize(v)
        return EmbeddingResult(v_norm, norm_before, v_norm.size)


def align_face_5pt(image: np.ndarray, src_kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)):
    """Similarity transform from detected 5 points to standard ArcFace points"""
    if src_kps.shape != (5, 2):
        return None, None

    # Enforce left/right order
    if src_kps[0, 0] > src_kps[1, 0]:
        src_kps[[0, 1]] = src_kps[[1, 0]]
    if src_kps[3, 0] > src_kps[4, 0]:
        src_kps[[3, 4]] = src_kps[[4, 3]]

    M, _ = cv2.estimateAffinePartial2D(src_kps, TARGET_POINTS, method=cv2.RANSAC)
    if M is None:
        return None, None

    aligned = cv2.warpAffine(image, M, out_size, flags=cv2.INTER_LINEAR)
    return aligned, M


def draw_text_block(img, lines, origin=(10, 30), scale=0.7, color=(0, 255, 0)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(28 * scale)


def draw_embedding_heatmap(img: np.ndarray, emb: np.ndarray, top_left=(10, 220), cell_scale=6):
    D = emb.size
    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))
    mat = np.zeros((rows, cols), dtype=np.float32)
    mat.flat[:D] = emb
    norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    gray = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (cols * cell_scale, rows * cell_scale), interpolation=cv2.INTER_NEAREST)

    x, y = top_left
    h, w = heat.shape[:2]
    ih, iw = img.shape[:2]
    if x + w > iw or y + h > ih:
        return
    img[y:y+h, x:x+w] = heat
    cv2.putText(img, "embedding heatmap", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error. Try index 1 or 2.")
        return

    # MediaPipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ONNX embedder
    try:
        embedder = ArcFaceEmbedderONNX(debug=True)
    except Exception as e:
        print(f"Failed to load ArcFace ONNX model:\n{e}")
        print("Make sure models/embedder_arcface.onnx exists.")
        return

    prev_emb: Optional[np.ndarray] = None
    print("Embedding demo. q = quit, p = print stats")

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        h, w = frame.shape[:2]

        # ── MediaPipe landmarks ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        aligned = None
        emb_result = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            idxs = [LEFT_EYE, RIGHT_EYE, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT]
            kps = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs], dtype=np.float32)

            aligned, _ = align_face_5pt(frame, kps)

            if aligned is not None:
                emb_result = embedder.embed(aligned)

        # ── Visualization ──
        info = []
        if emb_result:
            info.append(f"dim: {emb_result.dim}")
            info.append(f"norm (pre-L2): {emb_result.norm_before:.3f}")

            if prev_emb is not None:
                sim = float(np.dot(prev_emb, emb_result.embedding))
                info.append(f"cosine sim (prev): {sim:.3f}")

            prev_emb = emb_result.embedding

            # Small aligned preview
            aligned_small = cv2.resize(aligned, (160, 160))
            vis[10:170, w-170:w-10] = aligned_small

            # Heatmap
            draw_embedding_heatmap(vis, emb_result.embedding)

        else:
            info.append("No face detected")

        draw_text_block(vis, info, origin=(10, 30))

        # FPS
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ArcFace Embedding Demo", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p") and prev_emb is not None:
            print("\n[Current Embedding]")
            print("dim:", prev_emb.size)
            print("min / max:", prev_emb.min(), prev_emb.max())
            print("first 10:", prev_emb[:10])
            print("norm:", np.linalg.norm(prev_emb))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()