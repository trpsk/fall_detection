import cv2
from PIL import Image
import time
from collections import deque
import os
from fall_prediction import Fall_prediction
import numpy as np

CAMERA_SOURCE = 0
# Number of frames the prediction function expects (2 or 3)
FRAMES_FOR_PREDICTION = 2
# How often to run the analysis - in seconds
ANALYSIS_INTERVAL_SECONDS = 1
BUFFER_SIZE = FRAMES_FOR_PREDICTION

SAVE_FALL_IMAGES = True
FALL_IMAGE_SAVE_PATH = "detected_falls"

# Create directory for saving fall images if it doesn't exist
if SAVE_FALL_IMAGES:
    if not os.path.exists(FALL_IMAGE_SAVE_PATH):
        try:
            os.makedirs(FALL_IMAGE_SAVE_PATH)
            print(f"Created directory: {FALL_IMAGE_SAVE_PATH}")
        except OSError as e:
            print(f"Error creating directory {FALL_IMAGE_SAVE_PATH}: {e}")
            SAVE_FALL_IMAGES = False
    if SAVE_FALL_IMAGES:
        print(f"Detected fall images will be saved to: {FALL_IMAGE_SAVE_PATH}")

print(f"Attempting to open camera source: {CAMERA_SOURCE}")
cap = cv2.VideoCapture(CAMERA_SOURCE)

# Checking for camera opening, with Windows DirectShow fallback
if not cap.isOpened():
    print(f"Warning: Default camera backend failed. Trying DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(CAMERA_SOURCE + cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera source: {CAMERA_SOURCE}")
        exit()
print(f"Camera source {CAMERA_SOURCE} opened successfully.")

# Deques for efficient fixed-size frame buffering
frame_buffer_pil = deque(maxlen=BUFFER_SIZE) # Stores PIL Images for Fall_prediction
frame_buffer_cv = deque(maxlen=BUFFER_SIZE)  # Stores original OpenCV frames for saving/display
last_analysis_time = time.time()
response_from_prediction = None

try:
    while True:
        ret, cv_frame_original = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            time.sleep(0.5)
            continue

        try:
            rgb_frame = cv2.cvtColor(cv_frame_original, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
        except Exception as e:
            print(f"Error during frame conversion: {e}")
            continue

        # Adding current frames to the buffers
        frame_buffer_pil.append(pil_frame)
        frame_buffer_cv.append(cv_frame_original.copy())

        current_time = time.time()
        # Checking if enough time has passed and the buffer is full
        if current_time - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS and len(frame_buffer_pil) == BUFFER_SIZE:
            # Updating time immediately to prepare for next interval
            last_analysis_time = current_time

            # Getting the sequence of frames from buffers for analysis
            sequence_pil = list(frame_buffer_pil)
            sequence_cv = list(frame_buffer_cv)

            # Processing time
            processing_start_time = time.time()
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running fall detection...")

            img1_pil, img2_pil, img3_pil = None, None, None
            # Getting the last CV frame for potential saving
            cv_frame_at_detection = sequence_cv[-1] if sequence_cv else None

            # Calling the external Fall_prediction module
            try:
                # Handling 2 or 3 frame prediction based on config
                if FRAMES_FOR_PREDICTION == 3:
                    img1_pil, img2_pil, img3_pil = sequence_pil[0], sequence_pil[1], sequence_pil[2]
                    response_from_prediction = Fall_prediction(img1_pil, img2_pil, img3_pil)
                elif FRAMES_FOR_PREDICTION == 2:
                    img1_pil, img2_pil = sequence_pil[-2], sequence_pil[-1] # Get last two
                    response_from_prediction = Fall_prediction(img1_pil, img2_pil)
                else:
                    print(f"Warning: FRAMES_FOR_PREDICTION ({FRAMES_FOR_PREDICTION}) not supported.")
                    response_from_prediction = None
            except Exception as e:
                print(f"Error during Fall_prediction call: {e}")
                response_from_prediction = None # Ensure response is None on error

            processing_end_time = time.time()
            duration = processing_end_time - processing_start_time
            print(f"  Fall_prediction took: {duration:.3f} seconds")

            # Handling the dictionary returned by Fall_prediction
            if response_from_prediction:
                category = response_from_prediction.get('category', 'N/A')
                confidence = response_from_prediction.get('confidence', 0.0)
                print(f"    Category: {category}, Confidence: {confidence:.2f}")

                # Checking if the category indicates a fall
                if isinstance(category, str) and category.lower() == 'fall':
                    print("\n***************************")
                    print("    !!! FALL DETECTED !!!")
                    print("***************************\n")

                    # Save image if a fall was detected and saving is enabled
                    if SAVE_FALL_IMAGES and cv_frame_at_detection is not None:
                        timestamp_fall = time.strftime("%Y%m%d_%H%M%S")
                        fall_filename = os.path.join(FALL_IMAGE_SAVE_PATH, f"fall_{timestamp_fall}_conf{confidence:.2f}.jpg")
                        try:
                            cv2.imwrite(fall_filename, cv_frame_at_detection)
                            print(f"    Saved fall image: {fall_filename}")
                        except Exception as e:
                            print(f"    Error saving fall image: {e}")
                    # CUSTOM ALERT FUNCTION HERE
                    # Like send_telegram_alert(cv_frame_at_detection, response_from_prediction)

            else:
                print("  No definitive detection result or error in processing.")

        # Live feed
        display_frame = cv_frame_original.copy()
        status_text = "Live - Press Q to Quit"
        # Updating status text with the last detection result
        if response_from_prediction and isinstance(response_from_prediction.get('category'), str) :
            cat_disp = response_from_prediction.get('category', 'N/A')
            conf_disp = response_from_prediction.get('confidence', 0.0)
            status_text = f"Last: {cat_disp} ({conf_disp:.2f})" # Display last result
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, time.strftime('%Y-%m-%d %H:%M:%S'), (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Live Fall Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Exiting...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()