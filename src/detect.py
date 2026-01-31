import cv2

def main():
    # Load pre-trained Haar cascade for frontal faces
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")
    
    print("Haar face detect (minimal). Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        
        # Convert to grayscale (Haar needs gray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # how much size is reduced at each scale
            minNeighbors=5,       # how many neighbors each candidate rectangle should have
            minSize=(60, 60)      # minimum possible face size
        )
        
        # Draw rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow("Face Detection", frame)
        
        # Exit on 'q'
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()