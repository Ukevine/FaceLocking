import cv2

def main():
    cap = cv2.VideoCapture(0)           # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try changing index (0/1/2).")
    
    print("Camera test. Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        
        cv2.imshow("Camera Test", frame)
        
        # Wait 1 ms for key press; check if 'q' was pressed
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()