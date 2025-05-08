# Real-Time Fall Detection System (Modified)

This project provides a real-time fall detection system using OpenCV and TensorFlow Lite, adapted for use with a standard webcam feed on Windows.

## Based On

This project is a modified version based on the code from the Ambianic fall-detection repository:
<https://github.com/ambianic/fall-detection>

The original code is licensed under the Apache License 2.0. The full license text is included in the `LICENSE` file in this repository.

## Modifications

Key modifications from the original Ambianic code include:
*   Adaptation for real-time processing from a live camera feed (using OpenCV).
*   Removal of internal `time.sleep` calls from the prediction function for improved live performance.
*   Adjustments to fall detection parameters (e.g., angle threshold, confidence threshold) based on testing.
*   Bug fixes related to library imports and object attributes for compatibility with standard TensorFlow on Windows.
*   Added functionality to save an image upon fall detection.
*   Developed primarily for CPU execution on Windows (EdgeTPU support potentially disabled/removed in configuration).

## Setup (Windows)

1.  **Create Python Environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
2.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install opencv-python Pillow numpy PyYAML tensorflow matplotlib
    # Make sure TensorFlow (CPU or GPU version) is installed correctly
    ```

## Running

1.  Activate the virtual environment (`.\.venv\Scripts\Activate.ps1`).
2.  Run the live detection script:
    ```bash
    python live_fall_detector.py
    ```
3.  Press 'q' in the OpenCV window to quit. Detected fall images are saved in the `detected_falls` folder by default.

## Configuration

Key parameters can be adjusted within the Python scripts:
*   `CAMERA_SOURCE` in `live_fall_detector.py`
*   `ANALYSIS_INTERVAL_SECONDS` in `live_fall_detector.py`
*   `confidence_threshold` in `fall_prediction.py` -> `_fall_detect_config()`
*   `_fall_factor` in `src/pipeline/fall_detect.py` -> `FallDetector.__init__()`

## License

This modified project retains the **Apache License 2.0** inherited from the original Ambianic code. See the `LICENSE` file for details. Please ensure original copyright notices in source files are preserved.
