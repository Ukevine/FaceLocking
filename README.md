# Face Locking Feature – Intelligent Robotics Assignment


## Project Overview

This project extends the original CPU-only face recognition system (using ArcFace ONNX + 5-point landmark alignment) with a **Face Locking** feature.

**Face Locking** means:
- The system recognizes a specific enrolled person (target identity)
- Once locked, it tracks **only that person** across frames — even if other faces appear
- It detects and records simple face actions/movements in real time
- It maintains the lock during brief occlusions or recognition drops
- All observed actions are saved in a timestamped history file

## Features Implemented

- **Manual target selection** (hardcoded or via input – currently set to "Gabriel")
- **Locking** on the target identity when confidently recognized
- **Stable tracking** using IOU + embedding similarity + timeout-based unlock
- **Action detection** (while locked):
  - Face moved left
  - Face moved right
  - Eye blink (using Eye Aspect Ratio)
  - Smile (mouth wider than normal)
  - Laugh (mouth wide + vertically open)
- **Action history logging** to file in the format:  
  `<name>_history_<timestamp>.txt`  
  Example: `gabriel_history_1738339200123.txt`

Each line in the history file contains:
