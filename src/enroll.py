"""
enroll.py - Enrollment tool
Collects multiple aligned faces → computes mean ArcFace embedding per person
Run: python src/enroll.py
Controls:
  SPACE = capture one sample
  a     = toggle auto-capture
  s     = save enrollment (mean embedding)
  r     = reset new samples only
  q     = quit
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp

from .embed import ArcFaceEmbedderONNX, align_face_5pt  # assuming you have these from previous

# ── Config ──
@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")
    save_crops: bool = True
    crops_dir: Path = Path("data/enroll")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.4   # slower than 0.25 to avoid too many similar
    max_existing_crops: int = 200

# ── DB Helpers ──
def ensure_dirs(cfg: EnrollConfig):
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)

def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}

def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict):
    np.savez(cfg.out_db_npz, **{k: v for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("No embeddings to average")
    E = np.stack(embeddings, axis=0)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-10)
    return m.astype(np.float32)

# ── Load existing crops ──
def load_existing_samples(cfg: EnrollConfig, emb: ArcFaceEmbedderONNX, person_dir: Path) -> List[np.ndarray]:
    if not cfg.save_crops or not person_dir.exists():
        return []
    
    crops = sorted(person_dir.glob("*.jpg"))[-cfg.max_existing_crops:]
    embeddings = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            res = emb.embed(img)
            embeddings.append(res.embedding)
        except Exception as e:
            print(f"Failed to embed {p.name}: {e}")
    return embeddings

# ── UI ──
def draw_status(frame: np.ndarray, name: str, base_count: int, new_count: int, needed: int,
                auto: bool, msg: str = ""):
    total = base_count + new_count
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} / {needed}",
        f"Auto: {'ON' if auto else 'OFF'}  (a = toggle)",
        "SPACE = capture   s = save   r = reset new   q = quit",
    ]
    if msg:
        lines = [msg] + lines
    
    y = 35
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        y += 28

# ── Main ──
def main():
    cfg = EnrollConfig()
    ensure_dirs(cfg)

    name = input("Enter person name (e.g. Alice): ").strip()
    if not name:
        print("No name → exiting.")
        return

    person_dir = cfg.crops_dir / name

    # Pipeline components
    try:
        embedder = ArcFaceEmbedderONNX(debug=True)
    except Exception as e:
        print(f"Failed to load ArcFace model:\n{e}")
        return

    db = load_db(cfg)
    base_embeddings = load_existing_samples(cfg, embedder, person_dir)

    new_embeddings: List[np.ndarray] = []
    status_msg = f"Loaded {len(base_embeddings)} existing samples." if base_embeddings else ""

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed. Try index 1.")
        return

    cv2.namedWindow("enroll", cv2.WINDOW_NORMAL)
    cv2.namedWindow("aligned", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aligned", 240, 240)

    auto = False
    last_auto_time = 0.0

    print("\nEnrollment mode started.")
    print("Tip: vary angle, expression, lighting slightly.")
    print("SPACE = capture, a = auto, s = save, r = reset new, q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        h, w = frame.shape[:2]

        # Process face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        aligned = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            idxs = [33, 263, 1, 61, 291]  # left eye, right eye, nose, left mouth, right mouth
            kps = np.array([[lm[i].x * w, lm[i].y * h] for i in idxs], dtype=np.float32)

            aligned, _ = align_face_5pt(frame, kps)

        # Auto-capture
        now = time.time()
        if auto and aligned is not None and (now - last_auto_time) >= cfg.auto_capture_every_s:
            res = embedder.embed(aligned)
            new_embeddings.append(res.embedding)
            last_auto_time = now
            status_msg = f"Auto captured ({len(new_embeddings)} new)"

            if cfg.save_crops:
                ts = int(now * 1000)
                cv2.imwrite(str(person_dir / f"{ts}.jpg"), aligned)

        # Draw UI
        draw_status(vis, name, len(base_embeddings), len(new_embeddings),
                    cfg.samples_needed, auto, status_msg)

        if aligned is not None:
            cv2.imshow("aligned", aligned)
        else:
            cv2.imshow("aligned", np.zeros((112, 112, 3), np.uint8))

        cv2.imshow("enroll", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            auto = not auto
            status_msg = f"Auto {'ON' if auto else 'OFF'}"
        elif key == ord("r"):
            new_embeddings.clear()
            status_msg = "New samples reset (existing kept)"
        elif key == ord(" "):  # SPACE
            if aligned is None:
                status_msg = "No face → not captured"
            else:
                res = embedder.embed(aligned)
                new_embeddings.append(res.embedding)
                status_msg = f"Captured ({len(new_embeddings)} new)"
                if cfg.save_crops:
                    ts = int(time.time() * 1000)
                    cv2.imwrite(str(person_dir / f"{ts}.jpg"), aligned)
        elif key == ord("s"):
            total = len(base_embeddings) + len(new_embeddings)
            if total < 5:
                status_msg = f"Too few samples ({total}). Need at least 5–{cfg.samples_needed}"
                continue

            template = mean_embedding(base_embeddings + new_embeddings)
            db[name] = template

            meta = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dim": int(template.size),
                "identities": list(db.keys()),
                "samples_used": total,
                "note": "Mean L2-normalized ArcFace embedding. Use cosine similarity."
            }

            save_db(cfg, db, meta)
            status_msg = f"Saved {name} ({total} samples). DB has {len(db)} people now."
            print(status_msg)

            # Reload base (in case crops changed)
            base_embeddings = load_existing_samples(cfg, embedder, person_dir)
            new_embeddings.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()