
from __future__ import annotations
import datetime
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e


from .haar_5pt import align_face_5pt


TARGET_NAME = "dianah"  

# Action detection thresholds 
BLINK_EAR_THRESH = 0.22  
SMILE_MOUTH_WIDTH_RATIO = 1.4  
MOVE_DELTA_THRESH = 20 
MISSING_FRAMES_UNLOCK = 10  
SMILE_OPEN_THRESH = 0.15 

# History file
HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  
    eye_left_kps: np.ndarray 
    eye_right_kps: np.ndarray  
    mouth_kps: np.ndarray  

@dataclass
class LockedFace:
    name: str
    prev_center_x: float
    prev_ear: float
    prev_mouth_width_ratio: float
    prev_mouth_open: float
    prev_box: Tuple[int, int, int, int]
    embedding: np.ndarray
    missing_count: int = 0
    history_file: Optional[Path] = None

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def compute_ear(eye_kps: np.ndarray) -> float:
   
    v1 = np.linalg.norm(eye_kps[1] - eye_kps[5])
    v2 = np.linalg.norm(eye_kps[2] - eye_kps[4])
    h = np.linalg.norm(eye_kps[0] - eye_kps[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def compute_mouth_metrics(mouth_kps: np.ndarray, eye_dist: float) -> Tuple[float, float]:
    
    width = np.linalg.norm(mouth_kps[0] - mouth_kps[1])
    height = np.linalg.norm(mouth_kps[2] - mouth_kps[3])
    width_ratio = width / (eye_dist + 1e-6)
    open_ratio = height / width
    return width_ratio, open_ratio


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


class ArcFaceEmbedderONNX:
    def __init__(self, model_path: str = "models/embedder_arcface.onnx", input_size: Tuple[int, int] = (112, 112)):
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        return self._l2_normalize(y.reshape(-1))


class HaarFaceMeshExtended:
    def __init__(self, min_size: Tuple[int, int] = (70, 70)):
        haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar: {haar_xml}")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
       
        self.idx_5pt = [33, 263, 1, 61, 291]  
        
        self.idx_left_eye = [362, 385, 387, 263, 373, 380]
        
        self.idx_right_eye = [33, 160, 158, 133, 153, 144]
        
        self.idx_mouth = [61, 291, 13, 14]
        self.min_size = min_size

    def _extract_kps(self, lm, idxs, W, H):
        return np.array([[lm[i].x * W, lm[i].y * H] for i in idxs], dtype=np.float32)

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size)
        if len(faces) == 0:
            return []
        # Sorting by area
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        out = []
        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1 = max(0, int(x - mx)), max(0, int(y - my))
            rx2, ry2 = min(W, int(x + w + mx)), min(H, int(y + h + my))
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            rw, rh = roi.shape[1], roi.shape[0]
            kps5 = self._extract_kps(lm, self.idx_5pt, rw, rh)
            kps5[:, 0] += rx1
            kps5[:, 1] += ry1
            # Building bbox from 5pt
            x_min, y_min = np.min(kps5, axis=0)
            x_max, y_max = np.max(kps5, axis=0)
            bw = x_max - x_min
            bh = y_max - y_min
            x1 = int(max(0, x_min - 0.55 * bw))
            y1 = int(max(0, y_min - 0.85 * bh))
            x2 = int(min(W - 1, x_max + 0.55 * bw))
            y2 = int(min(H - 1, y_max + 1.15 * bh))
            # Extra kps
            left_eye = self._extract_kps(lm, self.idx_left_eye, rw, rh)
            left_eye[:, 0] += rx1
            left_eye[:, 1] += ry1
            right_eye = self._extract_kps(lm, self.idx_right_eye, rw, rh)
            right_eye[:, 0] += rx1
            right_eye[:, 1] += ry1
            mouth = self._extract_kps(lm, self.idx_mouth, rw, rh)
            mouth[:, 0] += rx1
            mouth[:, 1] += ry1
            out.append(FaceDet(x1, y1, x2, y2, 1.0, kps5, left_eye, right_eye, mouth))
        return out


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = dist_thresh
        self._names = sorted(db.keys())
        self._mat = np.stack([db[n] for n in self._names], axis=0) if self._names else None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._names = sorted(self.db.keys())
        self._mat = np.stack([self.db[n] for n in self._names], axis=0) if self._names else None

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None:
            return MatchResult(None, 1.0, 0.0, False)
        sims = np.dot(self._mat, emb.reshape(-1))
        best_i = np.argmax(sims)
        best_sim = sims[best_i]
        best_dist = 1.0 - best_sim
        accepted = best_dist <= self.dist_thresh
        name = self._names[best_i] if accepted else None
        return MatchResult(name, best_dist, best_sim, accepted)


def record_action(locked: LockedFace, action: str, desc: str = ""):
    if locked.history_file is None:
        ts = int(time.time() * 100)
        locked.history_file = HISTORY_DIR / f"{locked.name.lower()}_history_{ts}.txt"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now} {action} {desc}\n"
    with open(locked.history_file, "a") as f:
        f.write(line)


def main():
    db_path = Path("data/db/face_db.npz")
    det = HaarFaceMeshExtended(min_size=(70, 70))
    embedder = ArcFaceEmbedderONNX()
    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db, dist_thresh=0.34)
    if TARGET_NAME not in matcher.db:
        print(f"Target '{TARGET_NAME}' not enrolled. Enroll first.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    print(f"Face Locking for '{TARGET_NAME}'. q=quit, r=reload DB, +/- thresh, d=debug, l=force lock")
    t0 = time.time()
    frames = 0
    fps = None
    show_debug = False
    locked: Optional[LockedFace] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        vis = frame.copy()
        # FPS
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        selected_face = None
        if locked is None:
            # Finding target
            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                if mr.name == TARGET_NAME:
                    # Lock
                    eye_dist = np.linalg.norm(f.kps[0] - f.kps[1])
                    _, mouth_open = compute_mouth_metrics(f.mouth_kps, eye_dist)
                    mouth_width_ratio, _ = compute_mouth_metrics(f.mouth_kps, eye_dist)
                    locked = LockedFace(
                        TARGET_NAME,
                        (f.x1 + f.x2) / 2.0,
                        (compute_ear(f.eye_left_kps) + compute_ear(f.eye_right_kps)) / 2.0,
                        mouth_width_ratio,
                        mouth_open,
                        (f.x1, f.y1, f.x2, f.y2),
                        emb
                    )
                    record_action(locked, "LOCKED", "Face locked")
                    selected_face = f
                    break
        else:
            # Track locked face using IOU + embedding confirm
            best_iou = 0.0
            best_face = None
            best_emb_sim = 0.0
            for f in faces:
                cur_iou = iou(locked.prev_box, (f.x1, f.y1, f.x2, f.y2))
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_face = f
            if best_face and best_iou > 0.3:
                aligned, _ = align_face_5pt(frame, best_face.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                emb_sim = cosine_similarity(locked.embedding, emb)
                if emb_sim > 0.6:  
                    selected_face = best_face
                    locked.missing_count = 0
                    locked.embedding = 0.7 * locked.embedding + 0.3 * emb  # Smooth embedding
                else:
                    locked.missing_count += 1
            else:
                locked.missing_count += 1
            if locked.missing_count >= MISSING_FRAMES_UNLOCK:
                record_action(locked, "UNLOCKED", "Face lost")
                if locked.history_file:
                    print(f"History saved to {locked.history_file}")
                locked = None
        # Process actions if locked and selected
        if locked and selected_face:
            center_x = (selected_face.x1 + selected_face.x2) / 2.0
            delta_x = center_x - locked.prev_center_x
            if delta_x > MOVE_DELTA_THRESH:
                record_action(locked, "MOVE_RIGHT", f"Delta: {delta_x:.1f}")
            elif delta_x < -MOVE_DELTA_THRESH:
                record_action(locked, "MOVE_LEFT", f"Delta: {delta_x:.1f}")
            locked.prev_center_x = center_x
            ear_left = compute_ear(selected_face.eye_left_kps)
            ear_right = compute_ear(selected_face.eye_right_kps)
            avg_ear = (ear_left + ear_right) / 2.0
            if avg_ear < BLINK_EAR_THRESH and locked.prev_ear >= BLINK_EAR_THRESH:
                record_action(locked, "BLINK")
            locked.prev_ear = avg_ear
            eye_dist = np.linalg.norm(selected_face.kps[0] - selected_face.kps[1])
            mouth_width_ratio, mouth_open = compute_mouth_metrics(selected_face.mouth_kps, eye_dist)
            if mouth_width_ratio > SMILE_MOUTH_WIDTH_RATIO and locked.prev_mouth_width_ratio <= SMILE_MOUTH_WIDTH_RATIO:
                if mouth_open > SMILE_OPEN_THRESH:
                    record_action(locked, "LAUGH")
                else:
                    record_action(locked, "SMILE")
            locked.prev_mouth_width_ratio = mouth_width_ratio
            locked.prev_mouth_open = mouth_open
            locked.prev_box = (selected_face.x1, selected_face.y1, selected_face.x2, selected_face.y2)
            # Draw locked indicator
            cv2.rectangle(vis, (selected_face.x1, selected_face.y1), (selected_face.x2, selected_face.y2), (255, 0, 0), 3)
            cv2.putText(vis, f"LOCKED: {locked.name}", (selected_face.x1, selected_face.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Draw all faces (optional)
        for f in faces:
            if f != selected_face:
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 1)
        # Header
        header = f"Target: {TARGET_NAME} Locked: {'Yes' if locked else 'No'} Thr(dist): {matcher.dist_thresh:.2f}"
        if fps:
            header += f" FPS: {fps:.1f}"
        cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow("face_lock", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
        elif key == ord("+"):
            matcher.dist_thresh = min(1.20, matcher.dist_thresh + 0.01)
        elif key == ord("-"):
            matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.01)
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("l") and locked is None and faces:  # Force lock to first face
            f = faces[0]
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)
            if mr.accepted and mr.name == TARGET_NAME:
                eye_dist = np.linalg.norm(f.kps[0] - f.kps[1])
                mouth_width_ratio, mouth_open = compute_mouth_metrics(f.mouth_kps, eye_dist)
                locked = LockedFace(
                    TARGET_NAME,
                    (f.x1 + f.x2) / 2.0,
                    (compute_ear(f.eye_left_kps) + compute_ear(f.eye_right_kps)) / 2.0,
                    mouth_width_ratio,
                    mouth_open,
                    (f.x1, f.y1, f.x2, f.y2),
                    emb
                )
                record_action(locked, "LOCKED", "Forced lock")
    if locked and locked.history_file:
        print(f"History saved to {locked.history_file}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()