"""Fall detection pipe element."""
from .inference import TFInferenceEngine
from src.pipeline.pose_engine import PoseEngine
from src import DEFAULT_DATA_DIR
import logging
import math
import time
from PIL import Image, ImageDraw
from pathlib import Path

# Ensure logging is configured somewhere in your main script or here
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


class FallDetector():

    """Detects falls comparing two images spaced about 1-2 seconds apart."""
    def __init__(self,
                 model=None,
                 labels=None,
                 confidence_threshold=0.15, # Default seems low here, make sure your config overrides this
                 model_name=None,
                 **kwargs
                 ):
        """Initialize detector with config parameters."""
        # CHANGED: Pass the confidence threshold to TFInferenceEngine as well
        log.info(f"Initializing FallDetector. Configured confidence threshold: {confidence_threshold}")
        self._tfengine = TFInferenceEngine(
                        model=model,
                        labels=labels,
                        confidence_threshold=confidence_threshold)
        self.model_name = model_name

        # ADDED: Attempt to create the data directory if it doesn't exist
        self._sys_data_dir = Path(DEFAULT_DATA_DIR)
        try:
            self._sys_data_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Using data directory: {self._sys_data_dir}")
        except Exception as e:
            log.error(f"Could not create or access data directory {self._sys_data_dir}: {e}")

        self._prev_data = [None] * 2
        self.POSE_VAL = '_prev_pose_dix'
        self.TIMESTAMP = '_prev_time'
        self.THUMBNAIL = '_prev_thumbnail'
        self.LEFT_ANGLE_WITH_YAXIS = '_prev_left_angle_with_yaxis'
        self.RIGHT_ANGLE_WITH_YAXIS = '_prev_right_angle_with_yaxis'
        self.BODY_VECTOR_SCORE = '_prev_body_vector_score'

        _dix = {self.POSE_VAL: {}, # CHANGED: Initialize POSE_VAL as empty dict, not list
                self.TIMESTAMP: time.monotonic(),
                self.THUMBNAIL: None,
                self.LEFT_ANGLE_WITH_YAXIS: None,
                self.RIGHT_ANGLE_WITH_YAXIS: None,
                self.BODY_VECTOR_SCORE: 0
                }
        # CHANGED: Initialize history properly to avoid index errors later
        self._prev_data[0] = _dix.copy()
        self._prev_data[1] = _dix.copy()

        # ADDED: Error handling for PoseEngine initialization
        try:
            self._pose_engine = PoseEngine(self._tfengine, self.model_name)
        except Exception as e:
            log.error(f"Failed to initialize PoseEngine: {e}")
            raise

        # CHANGED: Reduced fall factor threshold based on debugging
        self._fall_factor = 30 # Angle threshold in degrees
        # CHANGED: Use the threshold passed during initialization and ensure float
        self.confidence_threshold = float(confidence_threshold)
        log.debug(f"FallDetector confidence threshold set to: {self.confidence_threshold}")

        # CHANGED: Ensure min/max times are floats
        self.min_time_between_frames = 1.0
        self.max_time_between_frames = 10.0

        self.LEFT_SHOULDER = 'left shoulder'
        self.LEFT_HIP = 'left hip'
        self.RIGHT_SHOULDER = 'right shoulder'
        self.RIGHT_HIP = 'right hip'
        self.fall_detect_corr = [self.LEFT_SHOULDER, self.LEFT_HIP,
                                 self.RIGHT_SHOULDER, self.RIGHT_HIP]

    def process_sample(self, **sample):
        """Detect objects in sample image."""
        log.debug("%s received new sample", self.__class__.__name__)
        # CHANGED: Check if image key exists
        if not sample or 'image' not in sample:
            log.warning("Empty sample or sample missing 'image' key received.")
            yield None
        else:
            try:
                image = sample['image']
                inference_result, thumbnail = self.fall_detect(image=image)
                # CHANGED: Use a different var name for converted result
                inference_result_converted = self.convert_inference_result(inference_result)
                inf_meta = {'display': 'Fall Detection'}
                processed_sample = {
                    'image': image, 'thumbnail': thumbnail,
                    'inference_result': inference_result_converted, # Use converted result
                    'inference_meta': inf_meta
                }
                yield processed_sample
            except Exception as e:
                # CHANGED: Log full exception and yield None
                log.exception('Error "%s" while processing sample. Dropping sample.', str(e))
                yield None

    def calculate_angle(self, p):
        """Calculate angle b/w two lines."""
        # CHANGED: Added input validation
        if not (p and len(p) == 2 and len(p[0]) == 2 and len(p[1]) == 2):
             log.error(f"Invalid input for calculate_angle: {p}")
             return 0
        x1, y1 = p[0][0]; x2, y2 = p[0][1]
        x3, y3 = p[1][0]; x4, y4 = p[1][1]
        try:
             theta1 = math.atan2(y2 - y1, x2 - x1)
             theta2 = math.atan2(y4 - y3, x4 - x3)
             delta_theta = theta2 - theta1
             while delta_theta <= -math.pi: delta_theta += 2 * math.pi
             while delta_theta > math.pi: delta_theta -= 2 * math.pi
             angle = abs(math.degrees(delta_theta))
             if angle > 180: angle = 360 - angle
        except Exception as e:
             # CHANGED: Added error handling
             log.error(f"Error calculating angle for points {p}: {e}")
             angle = 0
        return angle

    def is_body_line_motion_downward(self, left_angle_with_yaxis, right_angle_with_yaxis, inx):
        """Checks if torso angle relative to vertical has increased."""
        test = False
        # CHANGED: Use .get() for safer access to previous data
        prev_l_angle = self._prev_data[inx].get(self.LEFT_ANGLE_WITH_YAXIS)
        prev_r_angle = self._prev_data[inx].get(self.RIGHT_ANGLE_WITH_YAXIS)
        l_angle_increased = (left_angle_with_yaxis is not None and prev_l_angle is not None and left_angle_with_yaxis > prev_l_angle)
        r_angle_increased = (right_angle_with_yaxis is not None and prev_r_angle is not None and right_angle_with_yaxis > prev_r_angle)
        if l_angle_increased or r_angle_increased:
            test = True
        # REMOVED: Debug print related to is_body_line_motion_downward
        return test

    def find_keypoints(self, image):
        """Finds pose, calculates spinal vector score, handles rotation."""
        min_score = self.confidence_threshold
        rotations = [Image.ROTATE_270, Image.ROTATE_90]
        angle = 0
        pose = None
        poses, thumbnail, _ = self._pose_engine.detect_poses(image)
        # CHANGED: Handle case where PoseEngine returns no poses
        if not poses:
             log.warning("PoseEngine returned no poses.")
             return None, thumbnail, 0, {}
        width, height = thumbnail.size
        spinal_vector_score, pose_dix = self.estimate_spinal_vector_score(poses[0])

        initial_score = spinal_vector_score # Store initial score before rotation attempts

        while spinal_vector_score < min_score and rotations:
            log.debug(f"Initial score {spinal_vector_score:.2f} < {min_score}. Trying rotation.")
            angle = rotations.pop()
            try:
                transposed = image.transpose(angle)
                rotated_poses, _, _ = self._pose_engine.detect_poses(transposed)
                if rotated_poses:
                    spinal_vector_score, pose_dix = self.estimate_spinal_vector_score(rotated_poses[0])
                else:
                    spinal_vector_score, pose_dix = 0, {}
                log.debug(f"Rotated ({angle}) score: {spinal_vector_score:.2f}")
            except Exception as e:
                log.error(f"Error processing rotated image: {e}")
                spinal_vector_score, pose_dix = 0, {}

        # CHANGED: Improved logic to decide which pose result (original or rotated) to use
        if spinal_vector_score >= min_score:
             pose = poses[0]
             # REMOVED: Debug print related to using pose after rotation check
             if angle != 0:
                 log.info(f"Adjusting coordinates due to rotation angle {angle}")
                 corrected_pose_dix = {}
                 if angle == Image.ROTATE_90:
                     for key, point_yx in pose_dix.items():
                         original_x = point_yx[1]
                         original_y = height - point_yx[0]
                         corrected_pose_dix[key] = (original_y, original_x)
                 elif angle == Image.ROTATE_270:
                     for key, point_yx in pose_dix.items():
                         original_x = width - point_yx[1]
                         original_y = point_yx[0]
                         corrected_pose_dix[key] = (original_y, original_x)
                 pose_dix = corrected_pose_dix
        elif initial_score >= min_score:
             log.debug("Rotation did not yield better score, using original pose.")
             pose = poses[0]
             spinal_vector_score, pose_dix = self.estimate_spinal_vector_score(pose)
             angle = 0
             # REMOVED: Debug print related to using original pose
        else:
            log.debug(f"No pose detected with sufficient confidence (Highest score: {max(initial_score, spinal_vector_score):.2f}, Threshold: {min_score})")
            pose = None
            spinal_vector_score = 0
            pose_dix = {}

        return pose, thumbnail, spinal_vector_score, pose_dix

    def find_changes_in_angle(self, pose_dix, inx):
        """Find angle change for torso lines b/w current and previous frame."""
        left_angle = 0; right_angle = 0
        # CHANGED: Use .get() for safer access
        prev_pose = self._prev_data[inx].get(self.POSE_VAL, {})
        prev_leftLine_corr_exist = all(k in prev_pose for k in [self.LEFT_SHOULDER, self.LEFT_HIP])
        curr_leftLine_corr_exist = all(k in pose_dix for k in [self.LEFT_SHOULDER, self.LEFT_HIP])
        prev_rightLine_corr_exist = all(k in prev_pose for k in [self.RIGHT_SHOULDER, self.RIGHT_HIP])
        curr_rightLine_corr_exist = all(k in pose_dix for k in [self.RIGHT_SHOULDER, self.RIGHT_HIP])
        # CHANGED: Added try-except blocks
        if prev_leftLine_corr_exist and curr_leftLine_corr_exist:
            try:
                temp_left_vector = [[prev_pose[self.LEFT_SHOULDER], prev_pose[self.LEFT_HIP]],
                                    [pose_dix[self.LEFT_SHOULDER], pose_dix[self.LEFT_HIP]]]
                left_angle = self.calculate_angle(temp_left_vector)
                log.debug(f"Left shoulder-hip angle change vs frame {inx}: {left_angle:.2f}")
            except Exception as e: log.error(f"Error calculating left angle change vs frame {inx}: {e}")
        if prev_rightLine_corr_exist and curr_rightLine_corr_exist:
            try:
                temp_right_vector = [[prev_pose[self.RIGHT_SHOULDER], prev_pose[self.RIGHT_HIP]],
                                     [pose_dix[self.RIGHT_SHOULDER], pose_dix[self.RIGHT_HIP]]]
                right_angle = self.calculate_angle(temp_right_vector)
                log.debug(f"Right shoulder-hip angle change vs frame {inx}: {right_angle:.2f}")
            except Exception as e: log.error(f"Error calculating right angle change vs frame {inx}: {e}")
        return max(left_angle, right_angle)

    def assign_prev_records(self, pose_dix, left_angle_with_yaxis, right_angle_with_yaxis, now, thumbnail, current_body_vector_score):
        """Assign current frame's data to the history buffer."""
        # CHANGED: Ensure numeric or None
        l_angle = float(left_angle_with_yaxis) if left_angle_with_yaxis is not None else None
        r_angle = float(right_angle_with_yaxis) if right_angle_with_yaxis is not None else None
        curr_data = {
             self.POSE_VAL: pose_dix.copy(), self.TIMESTAMP: now, self.THUMBNAIL: thumbnail,
             self.LEFT_ANGLE_WITH_YAXIS: l_angle, self.RIGHT_ANGLE_WITH_YAXIS: r_angle,
             self.BODY_VECTOR_SCORE: float(current_body_vector_score) # Ensure float
        }
        self._prev_data[0] = self._prev_data[1]
        self._prev_data[1] = curr_data
        # REMOVED: Debug print related to updating history

    def draw_lines(self, thumbnail, pose_dix, score):
        """Draw body lines if available. Return number of lines drawn."""
        draw = ImageDraw.Draw(thumbnail)
        body_lines_drawn = 0
        # CHANGED: Check if pose_dix is None or empty
        if not pose_dix: return body_lines_drawn
        # CHANGED: Added helper for coordinate conversion and safety
        def to_xy(yx_tuple): return (int(yx_tuple[1]), int(yx_tuple[0])) if yx_tuple and len(yx_tuple) == 2 else None
        left_shoulder_xy = to_xy(pose_dix.get(self.LEFT_SHOULDER)); left_hip_xy = to_xy(pose_dix.get(self.LEFT_HIP))
        right_shoulder_xy = to_xy(pose_dix.get(self.RIGHT_SHOULDER)); right_hip_xy = to_xy(pose_dix.get(self.RIGHT_HIP))
        if left_shoulder_xy and left_hip_xy: draw.line([left_shoulder_xy, left_hip_xy], fill='red', width=2); body_lines_drawn += 1
        if right_shoulder_xy and right_hip_xy: draw.line([right_shoulder_xy, right_hip_xy], fill='blue', width=2); body_lines_drawn += 1
        if body_lines_drawn > 0:
             timestr = int(time.monotonic()*1000)
             debug_image_file_name = f'tmp-fall-detect-thumbnail-{timestr}-score-{score:.2f}.jpg'
             save_path = Path(self._sys_data_dir) / debug_image_file_name
             try:
                 img_to_save = thumbnail.copy() # CHANGED: Work on a copy
                 if img_to_save.mode != 'RGB': img_to_save = img_to_save.convert('RGB')
                 img_to_save.save(save_path, format='JPEG')
                 # REMOVED: Debug print related to saving thumbnail (can still use logging)
                 log.debug(f"Saved debug thumbnail to {save_path}") # Kept standard log.debug
             except Exception as e: log.error(f"Error saving debug thumbnail {save_path}: {e}")
        return body_lines_drawn

    def get_line_angles_with_yaxis(self, pose_dix):
        """Find angle b/w shoulder-hip line and yaxis."""
        # CHANGED: Handle empty pose_dix, improve vertical line def, add try-except
        if not pose_dix: return None, None
        height = getattr(self._pose_engine, '_tensor_image_height', 100)
        vertical_line = [(0, 0), (0, height)] # Uses (x, y)
        def to_xy(yx): return (yx[1], yx[0]) if yx else None
        left_shoulder_pt = pose_dix.get(self.LEFT_SHOULDER); left_hip_pt = pose_dix.get(self.LEFT_HIP)
        right_shoulder_pt = pose_dix.get(self.RIGHT_SHOULDER); right_hip_pt = pose_dix.get(self.RIGHT_HIP)
        l_angle = None; r_angle = None
        if left_shoulder_pt and left_hip_pt:
            try: l_angle = self.calculate_angle([vertical_line, [to_xy(left_shoulder_pt), to_xy(left_hip_pt)]])
            except Exception as e: log.error(f"Error calculating left angle with y-axis: {e}")
        if right_shoulder_pt and right_hip_pt:
             try: r_angle = self.calculate_angle([vertical_line, [to_xy(right_shoulder_pt), to_xy(right_hip_pt)]])
             except Exception as e: log.error(f"Error calculating right angle with y-axis: {e}")
        return l_angle, r_angle

    def estimate_spinal_vector_score(self, pose):
        """Estimates confidence score based on shoulder and hip keypoints."""
        pose_dix = {}; spinalVectorScore = 0; leftVectorScore = 0; rightVectorScore = 0
        # CHANGED: Added check for invalid pose object
        if not pose or not pose.keypoints: log.warning("estimate_spinal_vector_score received invalid pose object."); return 0, {}
        # CHANGED: Use .get() for safe access
        left_shoulder = pose.keypoints.get(self.LEFT_SHOULDER); left_hip = pose.keypoints.get(self.LEFT_HIP)
        right_shoulder = pose.keypoints.get(self.RIGHT_SHOULDER); right_hip = pose.keypoints.get(self.RIGHT_HIP)
        is_leftVector = False; is_rightVector = False
        # CHANGED: Check points exist before calculating score
        if left_shoulder and left_hip:
            leftVectorScore = min(left_shoulder.score, left_hip.score)
            if leftVectorScore >= self.confidence_threshold: is_leftVector = True; pose_dix[self.LEFT_SHOULDER] = left_shoulder.yx; pose_dix[self.LEFT_HIP] = left_hip.yx
        if right_shoulder and right_hip:
            rightVectorScore = min(right_shoulder.score, right_hip.score)
            if rightVectorScore >= self.confidence_threshold: is_rightVector = True; pose_dix[self.RIGHT_SHOULDER] = right_shoulder.yx; pose_dix[self.RIGHT_HIP] = right_hip.yx
        # Combine scores based on detected vectors
        if is_leftVector and is_rightVector: spinalVectorScore = (leftVectorScore + rightVectorScore) / 2.0
        elif is_leftVector: spinalVectorScore = leftVectorScore * 0.9
        elif is_rightVector: spinalVectorScore = rightVectorScore * 0.9
        log.debug(f"Estimated spinal vector score: {spinalVectorScore:.2f}")
        return spinalVectorScore, pose_dix

    def fall_detect(self, image=None):
        assert image
        log.debug("--- Starting fall_detect for new frame ---") # Kept standard log.debug
        start_time = time.monotonic()
        now = time.monotonic()
        lapse_vs_prev1 = now - self._prev_data[1][self.TIMESTAMP]

        # CHANGED: Return previous thumbnail when skipping
        if self._prev_data[1].get(self.POSE_VAL) and lapse_vs_prev1 < self.min_time_between_frames: # Use .get()
            log.debug(f"Frame too soon. Lapse: {lapse_vs_prev1:.2f}s < Min: {self.min_time_between_frames:.2f}s.")
            # REMOVED: Debug print for skipping frame
            return None, self._prev_data[1].get(self.THUMBNAIL)

        # CHANGED: Added try-except for find_keypoints
        try:
            pose, thumbnail, spinal_vector_score, pose_dix = self.find_keypoints(image)
            # REMOVED: Debug print for find_keypoints result
        except Exception as e:
            log.error(f"Error in find_keypoints: {e}")
            pose, thumbnail, spinal_vector_score, pose_dix = None, None, 0, {}

        inference_result = None
        # CHANGED: Initialize angles
        left_angle_with_yaxis, rigth_angle_with_yaxis = None, None

        if not pose:
            log.debug(f"No valid pose detected (score {spinal_vector_score:.2f} < threshold {self.confidence_threshold}).")
        else:
            inference_result = []
            current_body_vector_score = spinal_vector_score
            left_angle_with_yaxis, rigth_angle_with_yaxis = self.get_line_angles_with_yaxis(pose_dix)
            # REMOVED: Debug print for angles with Y-axis

            # CHANGED: Correct history indexing (t_idx) and added safety (.get)
            for t_idx, t_offset in enumerate([-1, -2]):
                prev_frame_data = self._prev_data[t_idx]
                lapse = now - prev_frame_data[self.TIMESTAMP]
                # REMOVED: Debug print for comparison details

                if not prev_frame_data.get(self.POSE_VAL) or lapse > self.max_time_between_frames:
                    log.debug(f"Skipping t{t_offset}: No valid pose or too old (Lapse: {lapse:.2f}s)")
                    # REMOVED: Debug print for skipping comparison
                    continue

                downward_motion_detected = self.is_body_line_motion_downward(left_angle_with_yaxis, rigth_angle_with_yaxis, inx=t_idx)

                if not downward_motion_detected:
                    log.debug(f"No downward motion detected compared to t{t_offset}.")
                    # REMOVED: Debug print for no downward motion
                    continue

                leaning_angle = self.find_changes_in_angle(pose_dix, inx=t_idx)
                # REMOVED: Debug print for downward motion + angle change

                leaning_probability = 1 if leaning_angle > self._fall_factor else 0
                # REMOVED: Debug print for leaning probability

                prev_body_vector_score = prev_frame_data[self.BODY_VECTOR_SCORE]
                fall_score = leaning_probability * (prev_body_vector_score + current_body_vector_score) / 2.0
                # REMOVED: Debug print for fall score calculation

                if fall_score >= self.confidence_threshold:
                    inference_result.append(('FALL', fall_score, leaning_angle, pose_dix))
                    log.info(f"Fall detected comparing to frame t{t_offset}: {inference_result[-1]}")
                    # REMOVED: Debug print for *** FALL DETECTED ***
                    break
                else:
                    log.debug(f"No fall detected vs t{t_offset} due to low score: {fall_score:.2f} < {self.confidence_threshold:.2f}")
                    # REMOVED: Debug print for score below threshold

        log.debug("Saving current pose/state for subsequent comparison.")
        # Ensure angles are None or float before saving
        l_angle_to_save = float(left_angle_with_yaxis) if left_angle_with_yaxis is not None else None
        r_angle_to_save = float(rigth_angle_with_yaxis) if rigth_angle_with_yaxis is not None else None
        # CHANGED: Corrected parameters passed to assign_prev_records
        self.assign_prev_records(pose_dix, l_angle_to_save,
                                 r_angle_to_save, now, thumbnail,
                                 spinal_vector_score)

        # Debug drawing can be controlled separately if needed
        # Standard logging level controls this now
        if log.getEffectiveLevel() <= logging.DEBUG:
             if thumbnail and pose_dix:
                  # CHANGED: Draw on a copy to avoid modifying the stored thumbnail
                  self.draw_lines(thumbnail.copy(), pose_dix, spinal_vector_score)

        log.debug("--- Finished fall_detect for this frame ---") # Kept standard log.debug
        return inference_result, thumbnail

    def convert_inference_result(self, inference_result):
        """Converts internal fall detection tuple to a structured dictionary list."""
        inf_json = []
        # CHANGED: Check if inference_result is not None before iterating
        if inference_result:
            for inf in inference_result:
                 try:
                      label, confidence, leaning_angle, pose_data = inf
                      log.info(f"Converting result: label={label}, confidence={confidence:.2f}, angle={leaning_angle:.2f}")
                      # CHANGED: More dynamic keypoint extraction, use .get()
                      kp_corr = { k: pose_data.get(k) for k in self.fall_detect_corr }
                      one_inf = {
                          'label': label, 'confidence': float(confidence),
                          'leaning_angle': float(leaning_angle), 'keypoint_corr': kp_corr
                      }
                      inf_json.append(one_inf)
                 except Exception as e:
                      log.error(f"Error converting inference result item {inf}: {e}")
        return inf_json # Return list (possibly empty)