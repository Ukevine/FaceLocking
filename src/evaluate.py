"""
evaluate.py - Threshold evaluation & suggestion
Uses all saved aligned crops in data/enroll/<name>/*.jpg
Run: python src/evaluate.py
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, align_face_5pt  # reuse from previous files

# ── Config ──
@dataclass
class EvalConfig:
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80
    target_far: float = 0.01          # 1% FAR
    thresholds: tuple = (0.10, 0.70, 0.005)  # start, stop, step for distance sweep
    require_size: tuple = (112, 112)


# ── Helpers ──
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity (assumes L2-normalized vectors)"""
    return 1.0 - float(np.dot(a.ravel(), b.ravel()))


def list_people(cfg: EvalConfig) -> List[Path]:
    if not cfg.enroll_dir.exists():
        raise FileNotFoundError(f"Enroll directory missing: {cfg.enroll_dir}")
    return sorted(p for p in cfg.enroll_dir.iterdir() if p.is_dir())


def load_embeddings_for_person(
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
    cfg: EvalConfig
) -> List[np.ndarray]:
    imgs = sorted(person_dir.glob("*.jpg"))[:cfg.max_imgs_per_person]
    embeddings = []
    
    for path in imgs:
        img = cv2.imread(str(path))
        if img is None or img.shape[:2] != cfg.require_size:
            continue
        try:
            res = embedder.embed(img)
            embeddings.append(res.embedding)
        except Exception as e:
            print(f"Failed embedding {path.name}: {e}")
    
    return embeddings


def pairwise_distances(embs_a: List[np.ndarray], embs_b: List[np.ndarray], same_person: bool) -> List[float]:
    dists = []
    if same_person:
        # Genuine: all unique pairs within class
        for i in range(len(embs_a)):
            for j in range(i + 1, len(embs_a)):
                dists.append(cosine_distance(embs_a[i], embs_a[j]))
    else:
        # Impostor: all cross pairs
        for ea in embs_a:
            for eb in embs_b:
                dists.append(cosine_distance(ea, eb))
    return dists


def describe(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:5d}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
        f"p05={np.percentile(arr, 5):.4f}  p50={np.percentile(arr, 50):.4f}  "
        f"p95={np.percentile(arr, 95):.4f}"
    )


def sweep_thresholds(genuine: np.ndarray, impostor: np.ndarray, cfg: EvalConfig):
    start, stop, step = cfg.thresholds
    thresholds = np.arange(start, stop + 1e-8, step)
    
    results = []
    for thr in thresholds:
        far = np.mean(impostor <= thr) if impostor.size > 0 else 0.0
        frr = np.mean(genuine > thr)  if genuine.size  > 0 else 0.0
        results.append((thr, far, frr))
    
    return results


def main():
    cfg = EvalConfig()
    
    try:
        embedder = ArcFaceEmbedderONNX(debug=False)
    except Exception as e:
        print(f"Cannot load ArcFace model:\n{e}")
        print("Check models/embedder_arcface.onnx exists.")
        return
    
    people_dirs = list_people(cfg)
    if not people_dirs:
        print("No enrolled people found in data/enroll/. Run enroll.py first.")
        return
    
    per_person: Dict[str, List[np.ndarray]] = {}
    for pdir in people_dirs:
        embs = load_embeddings_for_person(embedder, pdir, cfg)
        if len(embs) >= cfg.min_imgs_per_person:
            per_person[pdir.name] = embs
            print(f"Loaded {len(embs)} embeddings for {pdir.name}")
        else:
            print(f"Skipped {pdir.name}: only {len(embs)} valid crops (< {cfg.min_imgs_per_person})")
    
    if len(per_person) < 2:
        print("Need at least 2 people with enough samples to evaluate genuine vs impostor.")
        return
    
    names = sorted(per_person.keys())
    
    # ── Genuine distances (intra-class) ──
    genuine_dists = []
    for name in names:
        genuine_dists.extend(pairwise_distances(per_person[name], per_person[name], same_person=True))
    
    # ── Impostor distances (inter-class) ──
    impostor_dists = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_dists.extend(pairwise_distances(
                per_person[names[i]], per_person[names[j]], same_person=False
            ))
    
    genuine = np.array(genuine_dists)
    impostor = np.array(impostor_dists)
    
    print("\n" + "="*60)
    print("Distance Distributions (cosine distance = 1 - similarity)")
    print("-"*60)
    print(f"Genuine (same person):  {describe(genuine)}")
    print(f"Impostor (different):    {describe(impostor)}")
    print()
    
    results = sweep_thresholds(genuine, impostor, cfg)
    
    print("Threshold Sweep (distance threshold)")
    print("thr    FAR      FRR")
    print("-"*30)
    for thr, far, frr in results[::max(1, len(results)//15)]:  # show ~15 lines
        print(f"{thr:.3f}  {far*100:5.2f}%  {frr*100:5.2f}%")
    
    # Find best threshold (lowest FRR among those with FAR <= target)
    best = None
    for thr, far, frr in results:
        if far <= cfg.target_far:
            if best is None or frr < best[2]:
                best = (thr, far, frr)
    
    print("\n" + "="*60)
    if best:
        thr, far, frr = best
        sim_thr = 1.0 - thr
        print(f"Recommended threshold (target FAR ≤ {cfg.target_far*100:.1f}%):")
        print(f"  Cosine distance ≤ {thr:.3f}")
        print(f"  OR Cosine similarity ≥ {sim_thr:.3f}")
        print(f"  → FAR: {far*100:.2f}%   FRR: {frr*100:.2f}%")
    else:
        print(f"No threshold meets FAR ≤ {cfg.target_far*100:.1f}% in sweep range.")
        print("Try:")
        print("  - Collecting more varied samples")
        print("  - Widening threshold range in EvalConfig")
        print("  - Lowering target_far if acceptable")
    
    print("\nUse this threshold in recognize.py for live matching.")


if __name__ == "__main__":
    main()