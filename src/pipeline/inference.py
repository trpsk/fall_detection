"""Tensorflow inference engine wrapper."""
import logging
import os
import numpy as np
import tensorflow as tf # CHANGED: Use main TensorFlow import

# CHANGED: Removed direct imports from tensorflow.lite or tflite_runtime
# Interpreter and experimental.load_delegate are accessed via tf.lite

log = logging.getLogger(__name__)


# CHANGED: Renamed parameter and added more robust delegate loading logic
def _get_edgetpu_interpreter(model_path_edgetpu=None):
    """
    Attempts to load a TFLite model with an EdgeTPU delegate.
    Returns an Interpreter object or None if it fails.
    """
    tf_interpreter = None
    # ADDED: Check if path is valid before proceeding
    if model_path_edgetpu and os.path.isfile(model_path_edgetpu):
        try:
            # ADDED: Basic OS check for delegate library name (simplification)
            delegate_name = ''
            if os.name == 'posix': # Linux-like
                delegate_name = 'libedgetpu.so.1.0'
            # elif os.name == 'nt': # Placeholder for Windows delegate
            #     delegate_name = 'edgetpu.dll'

            if delegate_name:
                # CHANGED: Use tf.lite.experimental.load_delegate
                edgetpu_delegate = tf.lite.experimental.load_delegate(delegate_name)
                if edgetpu_delegate:
                    # CHANGED: Use tf.lite.Interpreter
                    tf_interpreter = tf.lite.Interpreter(
                        model_path=model_path_edgetpu,
                        experimental_delegates=[edgetpu_delegate]
                    )
                    log.debug(f'EdgeTPU delegate loaded. Will use EdgeTPU model: {model_path_edgetpu}')
                else:
                    log.debug(f"Could not load EdgeTPU delegate: {delegate_name}")
            else:
                log.debug("EdgeTPU delegate name not specified for this OS or model not provided.")

        except Exception as e:
            # ADDED: More detailed logging on exception
            log.debug(f'EdgeTPU init error (this is expected if no EdgeTPU or wrong OS for delegate): {e}')
            log.debug(f'Exception type: {type(e).__name__}')
    else:
        log.debug("No valid EdgeTPU model path provided to _get_edgetpu_interpreter.")

    return tf_interpreter


class TFInferenceEngine:
    """Thin wrapper around TFLite Interpreter.

    Dynamically attempts to detect and use EdgeTPU if configured and available,
    otherwise falls back to TFLite CPU runtime.
    """

    def __init__(self,
                 model=None, # model is a dict: {'tflite': 'path', 'edgetpu': 'optional_path'}
                 labels=None,
                 confidence_threshold=0.8,
                 **kwargs
                 ):
        # ADDED: More robust input validation
        assert model and isinstance(model, dict), "Model configuration dictionary is required."
        assert 'tflite' in model and model['tflite'], 'TFLite AI model path (model["tflite"]) is required.'

        model_tflite_path = model['tflite']
        # ADDED: Use f-string in assertion message
        assert os.path.isfile(model_tflite_path), \
            f'TFLite AI model file does not exist: {model_tflite_path}'
        self._model_tflite_path = model_tflite_path

        model_edgetpu_path = model.get('edgetpu', None)
        # ADDED: Ensure edgetpu path is a valid file if provided, otherwise it's None
        self._model_edgetpu_path = model_edgetpu_path if model_edgetpu_path and os.path.isfile(model_edgetpu_path) else None

        assert labels, 'AI model labels path required.'
        # ADDED: Use f-string in assertion message
        assert os.path.isfile(labels), \
            f'AI model labels file does not exist: {labels}'
        self._model_labels_path = labels
        self._confidence_threshold = confidence_threshold

        self._tf_interpreter = None
        # ADDED: Clearer logic for attempting EdgeTPU first, then fallback
        if self._model_edgetpu_path:
            log.debug(f"Attempting to load EdgeTPU model: {self._model_edgetpu_path}")
            self._tf_interpreter = _get_edgetpu_interpreter(model_path_edgetpu=self._model_edgetpu_path)

        # If EdgeTPU interpreter wasn't successfully created (or not specified), fallback to CPU TFLite
        if not self._tf_interpreter:
            if self._model_edgetpu_path: # Log if we tried EdgeTPU and failed
                 log.debug(f"Failed to load EdgeTPU model or delegate. Falling back to TFLite CPU.")
            log.debug(f"Using TFLite CPU runtime with model: {self._model_tflite_path}")
            # ADDED: try-except around CPU interpreter creation
            try:
                # CHANGED: Use tf.lite.Interpreter
                self._tf_interpreter = tf.lite.Interpreter(model_path=self._model_tflite_path)
            except Exception as e:
                log.error(f"Failed to initialize TFLite CPU interpreter: {e}")
                raise RuntimeError(f"Could not initialize TFLite CPU interpreter: {e}") from e

        assert self._tf_interpreter, "Fatal: Failed to initialize any TFLite interpreter."

        self._tf_interpreter.allocate_tensors()

        # CHANGED: Store details directly in private attributes first
        self._tf_input_details = self._tf_interpreter.get_input_details()
        self._tf_output_details = self._tf_interpreter.get_output_details()

        # CHANGED: Check existence of details and use private attribute directly
        #          to fix the 'has no attribute input_details' error during init.
        if self._tf_input_details and len(self._tf_input_details) > 0:
            self._tf_is_quantized_model = \
                self._tf_input_details[0]['dtype'] != np.float32
        else:
            log.error("Could not get valid input details from TFLite interpreter!")
            self._tf_is_quantized_model = False
            # Consider: raise RuntimeError("Failed to get TFLite input details.")

    # ADDED: Properties made slightly more robust with hasattr checks
    @property
    def input_details(self):
        if not hasattr(self, '_tf_input_details'):
            log.warning("Accessing input_details before _tf_input_details is initialized.")
            return None
        return self._tf_input_details

    @property
    def output_details(self):
        if not hasattr(self, '_tf_output_details'):
            log.warning("Accessing output_details before _tf_output_details is initialized.")
            return None
        return self._tf_output_details

    @property
    def is_quantized(self):
        if not hasattr(self, '_tf_is_quantized_model'):
            log.warning("Accessing is_quantized before _tf_is_quantized_model is initialized.")
            return False # Default
        return self._tf_is_quantized_model

    @property
    def confidence_threshold(self):
        return self._confidence_threshold

    # ADDED: Checks for interpreter existence in methods
    def infer(self):
        """Invoke model inference on current input tensor."""
        if not self._tf_interpreter:
            log.error("Interpreter not initialized. Cannot run inference.")
            return None
        return self._tf_interpreter.invoke()

    def set_tensor(self, index=None, tensor_data=None):
        """Set tensor data at given reference index."""
        if not self._tf_interpreter:
            log.error("Interpreter not initialized. Cannot set tensor.")
            return
        assert isinstance(index, int)
        self._tf_interpreter.set_tensor(index, tensor_data)

    def get_tensor(self, index=None):
        """Return tensor data at given reference index."""
        if not self._tf_interpreter:
            log.error("Interpreter not initialized. Cannot get tensor.")
            return None
        assert isinstance(index, int)
        return self._tf_interpreter.get_tensor(index)