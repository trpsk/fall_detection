import os
import time
from src.pipeline.fall_detect import FallDetector # Make sure logging is set up in fall_detect or here

# Setup basic logging if not done elsewhere (optional but good practice)
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

def _fall_detect_config():
    """Loads configuration, ensuring paths are correct."""
    # ADDED: try-except block for robust config loading
    try:
        # ADDED: _dir calculation to make paths absolute/relative to this file
        _dir = os.path.dirname(os.path.abspath(__file__))
        _good_tflite_model = os.path.join(
            _dir,
            'ai_models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
        )
        # REMOVED/COMMENTED OUT: EdgeTPU model path to ensure CPU execution on systems without EdgeTPU
        # _good_edgetpu_model = os.path.join(
        #     _dir,
        #     'ai_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite'
        # )
        # CHANGED: Made labels path absolute/relative to this file using os.path.join
        _good_labels = os.path.join(
            _dir,
            'ai_models/pose_labels.txt'
        )

        # ADDED: Ensure model and label files exist before proceeding
        if not os.path.isfile(_good_tflite_model):
            raise FileNotFoundError(f"TFLite model not found at: {_good_tflite_model}")
        if not os.path.isfile(_good_labels):
             raise FileNotFoundError(f"Labels file not found at: {_good_labels}")

        config = {
            'model': {
                'tflite': _good_tflite_model,
                # 'edgetpu': _good_edgetpu_model, # REMOVED/COMMENTED OUT: Ensure EdgeTPU is not passed
            },
            'labels': _good_labels,
            'top_k': 3, # This parameter might be used by PoseEngine, check its usage
            # CHANGED: Lowered confidence threshold based on debugging (adjust as needed)
            'confidence_threshold': 0.3, # Example: Lowered threshold for testing
            'model_name':'mobilenet'
        }
        return config
    except Exception as e:
        # log.error(f"Error loading configuration: {e}") # Use logging if configured
        # ADDED: Print critical error if config fails
        print(f"CRITICAL ERROR loading configuration: {e}")
        raise # Re-raise the exception to stop execution if config fails

def Fall_prediction(img_1, img_2, img_3=None):
    """
    Processes a sequence of 2 or 3 images to detect falls.
    Internal time sleeps are removed for live feed processing.
    """
    # ADDED: try-except block for configuration loading
    try:
        config = _fall_detect_config()
        # ADDED: Ensure confidence_threshold is float if passed via config
        config['confidence_threshold'] = float(config.get('confidence_threshold', 0.6)) # Default fallback if missing
    except Exception as e:
         print(f"Error getting configuration in Fall_prediction: {e}")
         return None # Cannot proceed without config

    # CHANGED: Renamed variable for clarity
    result_dict = None
    # CHANGED: Use a list to store raw results from FallDetector
    final_result_list = []

    # ADDED: try-except block around the main processing logic
    try:
        # Instantiate FallDetector (original logic)
        fall_detector = FallDetector(**config)

        # ADDED: Helper function to handle the generator output from process_sample
        def process_detector_output(output_generator):
            nonlocal final_result_list # Modify the list in the outer scope
            try:
                 # FallDetector.process_sample yields results
                 for res_dict in output_generator:
                     if res_dict and 'inference_result' in res_dict:
                         # Store the raw list of detections (usually empty or one item)
                         final_result_list = res_dict['inference_result']
                         break # Assuming we only need the first yielded result per sample
                     else:
                         # Handle cases where generator yields None or unexpected dict
                         final_result_list = []
            except Exception as e:
                 print(f"Error processing detector output: {e}")
                 final_result_list = [] # Reset on error

        # --- Process Frame 1 ---
        process_detector_output(fall_detector.process_sample(image=img_1))

        # --- Process Frame 2 ---
        # REMOVED: time.sleep(fall_detector.min_time_between_frames)
        process_detector_output(fall_detector.process_sample(image=img_2))

        # CHANGED: Check result list structure and use .get() for safety
        if final_result_list and len(final_result_list) == 1:
             res = final_result_list[0]
             result_dict = {
                 "category": res.get('label'), # Use .get()
                 "confidence": res.get('confidence'), # Use .get()
                 "angle": res.get('leaning_angle'), # Use .get()
                 "keypoint_corr": res.get('keypoint_corr') # Use .get()
             }
             return result_dict # Return immediately if found after 2 frames

        # --- Process Frame 3 (only if needed and provided) ---
        # CHANGED: Added check to ensure we only process img_3 if no result yet
        if img_3 and (not final_result_list or len(final_result_list) != 1):
            # REMOVED: time.sleep(fall_detector.min_time_between_frames)
            process_detector_output(fall_detector.process_sample(image=img_3))

            # CHANGED: Check result list structure and use .get() for safety
            if final_result_list and len(final_result_list) == 1:
                 res = final_result_list[0]
                 result_dict = {
                     "category": res.get('label'), # Use .get()
                     "confidence": res.get('confidence'), # Use .get()
                     "angle": res.get('leaning_angle'), # Use .get()
                     "keypoint_corr": res.get('keypoint_corr') # Use .get()
                 }
                 return result_dict # Return if found after 3 frames

    except Exception as e:
        # log.exception("Error during Fall_prediction execution") # Use logging if configured
        # ADDED: Catch execution errors and print message
        print(f"Error during Fall_prediction execution: {e}")
        return None # Return None on error

    # If no single definitive result was found after all relevant frames
    # CHANGED: Return the clearer variable name
    return result_dict # Will be None if no fall was detected meeting criteria