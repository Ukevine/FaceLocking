"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceMesh (full-frame) -> extract 5 keypoints -> draw
Run:
python src/landmarks.py
Keys: q → quit
"""

import cv2
import numpy as np
import mediapipe as mp

# 5-point indices from MediaPipe FaceMesh (refined landmarks enabled)
IDX_LEFT_EYE    = 33
IDX_RIGHT_EYE   = 263
IDX_NOSE_TIP    = 1
IDX_MOUTH_LEFT  = 61
IDX_MOUTH_RIGHT = 291

def main():
    # Load Haar cascade for initial face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,           # important for iris/eye landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try changing index to 1 or 2.")

    print("Haar + MediaPipe 5pt landmarks. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar detection (rough bounding box)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw all detected Haar boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # MediaPipe FaceMesh on full RGB frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]

            pts = []
            for i in idxs:
                p = landmarks[i]
                pts.append([p.x * W, p.y * H])

            kps = np.array(pts, dtype=np.float32)  # shape (5, 2)

            # Enforce correct left/right ordering (in case of flip or error)
            if kps[0, 0] > kps[1, 0]:  # left eye x > right eye x
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:  # left mouth x > right mouth x
                kps[[3, 4]] = kps[[4, 3]]

            # Draw the 5 keypoints
            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 5, (0, 0, 255), -1)  # red for visibility

            cv2.putText(frame, "5pt Landmarks", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("5pt Landmarks (Haar + MediaPipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()