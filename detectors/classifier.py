# ------------------------------------------------------------------------------
# Classifier Module for Image Classification (ONNX Runtime)
# detectors/classifier.py
# ------------------------------------------------------------------------------
import os
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import requests
import json

from config import load_config
config = load_config()
from logging_config import get_logger
logger = get_logger(__name__)


# --- Define Constants to model paths ---
GITHUB_REPO_USER = "arminfabritzek"
GITHUB_REPO_NAME = "WatchMyBirds-Classifier"
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO_USER}/{GITHUB_REPO_NAME}/main"
LATEST_MODELS_POINTER_URL = f"{GITHUB_BASE_URL}/models/latest_models.json"

class ImageClassifier:
    def __init__(self, model_type=None, version_timestamp=None, model_path=None, class_path=None):
        """
        Initializes the ImageClassifier.

        Allows specifying model type, a specific version timestamp for download,
        or direct paths to local model/class files. Configuration provides defaults.

        Args:
            model_type (str, optional): Specific model sub-type (e.g., 'efficientnet_b0').
                                       Overrides config['CLASSIFIER_MODEL'] if provided.
            version_timestamp (str, optional): Specific 'YYYYMMDD_HHMMSS' timestamp to download.
                                              If None, the latest version for the model_type is used.
                                              Only relevant if downloading is enabled.
            model_path (str, optional): Direct path to a local ONNX model file. If provided,
                                       type/timestamp/download logic is skipped.
            class_path (str, optional): Direct path to a local class file. If provided with
                                       model_path, this is used. If None but model_path is
                                       provided, attempts to infer (e.g., 'model.txt' next to 'model.onnx').
        """
        self.config = config
        self.ort_session = None
        self.classes = []
        self.transform = None
        self.model_path = None
        self.class_path = None
        self.model_type = None  # Will be set based on logic below

        # --- Priority 1: Custom Paths Provided ---
        if model_path:
            logger.info(f"Using custom model path provided: {model_path}")
            if not os.path.exists(model_path):
                 logger.error(f"Custom model path specified but file not found: {model_path}")
                 raise FileNotFoundError(f"Custom model path file not found: {model_path}")
            self.model_path = model_path
            self.model_type = "custom_path"  # Indicate how model was specified

            if class_path:
                if not os.path.exists(class_path):
                    logger.warning(f"Custom class path specified but file not found: {class_path}. Class names will be default.")
                    self.class_path = None  # Treat as missing if not found
                else:
                    self.class_path = class_path
                    logger.info(f"Using custom class path provided: {class_path}")
            else:
                # Try to infer class path (.txt next to .onnx)
                base, _ = os.path.splitext(self.model_path)
                potential_class_path = base + ".txt"
                if os.path.exists(potential_class_path):
                    self.class_path = potential_class_path
                    logger.info(f"Inferred custom class path: {self.class_path}")
                else:
                    self.class_path = None  # Indicate missing class path
                    logger.warning(f"Custom model path provided, but no custom class path given or inferred (.txt). Class names will be default.")

            # Load directly from custom paths
            self._load_session_and_classes()  # Helper function loads session, classes, transform
            logger.info(f"ImageClassifier initialized using custom paths.")
            return  # Skip rest of init

        # --- Priority 2 & 3: Use Model Type (from arg or config) ---
        # Determine Model Type
        if model_type:
            self.model_type = model_type
            logger.info(f"Using model type specified in argument: {self.model_type}")
        else:
            self.model_type = self.config.get("CLASSIFIER_MODEL")
            if self.model_type:
                logger.info(f"Using model type from config: {self.model_type}")
            else:
                 logger.error("Model type must be specified via argument or 'CLASSIFIER_MODEL' config if custom paths are not used.")
                 raise ValueError("Model type not specified.")

        # Determine Local Paths based on Type
        default_cache_dir = "/models"
        base_local_path = self.config.get("CLASSIFIER_BASE_PATH", default_cache_dir)
        self.specific_model_dir = os.path.join(base_local_path, self.model_type)
        self.model_path = os.path.join(self.specific_model_dir, "classifier_best.onnx")
        self.class_path = os.path.join(self.specific_model_dir, "classifier_classes.txt")
        logger.info(f"Expecting model file for type '{self.model_type}' at: {self.model_path}")
        logger.info(f"Expecting class file for type '{self.model_type}' at: {self.class_path}")

        # Ensure local directory exists
        try:
            os.makedirs(self.specific_model_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create local model directory {self.specific_model_dir}: {e}")
            raise  # Cannot proceed without storage directory

        # Handle Downloading (if enabled in config)
        download_enabled = self.config.get("CLASSIFIER_DOWNLOAD_LATEST_MODEL", False)  # Default False if missing
        target_timestamp = None  # The specific version we aim for

        if download_enabled:
            if version_timestamp:  # User requested a SPECIFIC version
                logger.info(f"Download enabled and specific version requested: {version_timestamp}")
                target_timestamp = version_timestamp
            else:  # User wants the LATEST for the model type
                logger.info(f"Download enabled. Attempting to find latest version for '{self.model_type}'...")
                target_timestamp = self._get_latest_timestamp_for_type(self.model_type)

            if target_timestamp:
                logger.info(f"Targeting version timestamp: {target_timestamp}")
                # Check local existence BEFORE downloading
                model_exists_locally = os.path.exists(self.model_path)
                classes_exist_locally = os.path.exists(self.class_path)

                if not model_exists_locally:
                    logger.info(f"Local model file missing. Attempting download for timestamp {target_timestamp}...")
                    if self.download_model(target_timestamp):
                        # If model download succeeded, *now* check if classes need downloading
                        if not classes_exist_locally:
                             logger.info(f"Model downloaded successfully. Class file missing. Attempting class download...")
                             self.download_classes(target_timestamp)  # Ignore return, _load_classes handles missing file
                    else:
                        logger.error(f"Download failed for model version {target_timestamp}. Cannot proceed with loading this version.")
                        raise FileNotFoundError(f"Failed to download required model file for version {target_timestamp} to {self.model_path}")
                elif not classes_exist_locally:  # Model exists, classes missing
                     logger.info(f"Model file exists locally. Class file missing. Attempting class download for timestamp {target_timestamp}...")
                     self.download_classes(target_timestamp)  # Ignore return, _load_classes handles missing file
                else:
                    logger.info(f"Model and class files for version {target_timestamp} already exist locally. Skipping download.")
            else:
                logger.warning(f"Could not determine a version timestamp for download (Type='{self.model_type}', SpecificVersion='{version_timestamp}'). Relying on existing local files.")
        else:
             logger.info("Automatic download disabled. Relying on existing local files at expected paths.")

        # Load session and classes from final determined local paths
        self._load_session_and_classes()
        logger.info(f"ImageClassifier for '{self.model_type}' initialized.")
        # End of __init__

    def _load_session_and_classes(self):
        """Helper function to load ONNX session, classes, and transforms."""
        logger.debug("Loading ONNX session, classes, and transforms...")

        # --- Load Model ---
        if not self.model_path or not os.path.exists(self.model_path):
             logger.error(f"ONNX model file cannot be loaded. Path not set or file not found: {self.model_path}")
             raise FileNotFoundError(f"Required ONNX model file not found or path invalid: {self.model_path}")
        try:
            providers = ['CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                 providers.insert(0, 'CUDAExecutionProvider')
            elif 'ROCMExecutionProvider' in available_providers:
                 providers.insert(0, 'ROCMExecutionProvider')
            logger.info(f"Attempting to load ONNX session '{os.path.basename(self.model_path)}' with providers: {providers}")
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"Successfully loaded ONNX session from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX Runtime session from {self.model_path}: {e}")
            raise

        # --- Load Classes ---
        # Handle missing class file path when using a custom model path
        self.classes = self._load_classes()  # _load_classes handles None path or file not found

        # --- Get Transforms ---
        try:
            # Image size might not be in config if using custom path, handle this
            self.CLASSIFIER_IMAGE_SIZE = self.config.get("CLASSIFIER_IMAGE_SIZE", 224)  # Default if missing - ! Can cause errors !
        except Exception as e:
            logger.warning(f"Could not read CLASSIFIER_IMAGE_SIZE from config, using default 224. Error: {e}")
            self.CLASSIFIER_IMAGE_SIZE = 224
        self.transform = self._get_transform()

    def _get_latest_timestamp_for_type(self, requested_model_type):
        """Downloads latest_models.json and finds the timestamp for the requested model type."""
        logger.debug(f"Fetching latest model pointers from: {LATEST_MODELS_POINTER_URL}")
        try:
            response = requests.get(LATEST_MODELS_POINTER_URL, timeout=10)  # Short timeout for small JSON file
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            latest_models_data = response.json()  # Directly parse JSON response

            if not isinstance(latest_models_data, dict):
                logger.error(f"Invalid format in pointer file: Expected a JSON object, got {type(latest_models_data)}")
                return None

            timestamp = latest_models_data.get(requested_model_type)  # Use .get for safer lookup

            if timestamp:
                logger.info(f"Found latest timestamp for type '{requested_model_type}': {timestamp}")
                return str(timestamp)
            else:
                logger.warning(f"Timestamp for model type '{requested_model_type}' not found in {LATEST_MODELS_POINTER_URL}")
                logger.warning(f"Available types in pointer file: {list(latest_models_data.keys())}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred while fetching pointer file from {LATEST_MODELS_POINTER_URL}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching pointer file from {LATEST_MODELS_POINTER_URL}: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from pointer file {LATEST_MODELS_POINTER_URL}. Content: {response.text[:200]}...")
            return None
        except Exception as e:
             logger.error(f"An unexpected error occurred while getting the latest timestamp for type '{requested_model_type}': {e}")
             return None


    def download_model(self, version_timestamp):
        """Downloads the ONNX model for a specific version timestamp to the determined local path."""
        url = f"{GITHUB_BASE_URL}/models/{version_timestamp}/classifier_best.onnx"
        logger.info(f"Attempting to download ONNX model for '{self.model_type}' version '{version_timestamp}' from: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)  # Longer timeout for potentially large model file
            response.raise_for_status()
            # Download in chunks to handle large files
            with open(self.model_path, 'wb') as f:
                 for chunk in response.iter_content(chunk_size=8192):
                     f.write(chunk)
            logger.info(f"Successfully downloaded ONNX model to: {self.model_path}")
            return True  # Indicate success
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred while downloading model from {url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download ONNX model from {url}: {e}")
            return False
        except OSError as e:
             logger.error(f"Failed to save downloaded ONNX model to {self.model_path}: {e}")
             return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during model download: {e}")
            return False


    def download_classes(self, version_timestamp):
        """Downloads the classes file for a specific version timestamp TO the determined local path."""
        url = f"{GITHUB_BASE_URL}/models/{version_timestamp}/classifier_classes.txt"
        logger.info(f"Attempting to download classes file for '{self.model_type}' version '{version_timestamp}' from: {url}")
        try:
            response = requests.get(url, timeout=10)  # Shorter timeout for small text file
            response.raise_for_status()
            # Ensure content is decoded correctly (assume utf-8) before writing
            response.encoding = response.apparent_encoding or 'utf-8'  # Guess encoding or default to UTF-8
            with open(self.class_path, 'w', encoding=response.encoding) as f:
                f.write(response.text)
            logger.info(f"Successfully downloaded classifier classes file to: {self.class_path}")
            return True  # Indicate success
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred while downloading classes file from {url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download classifier classes file from {url}: {e}")
            return False
        except OSError as e:
             logger.error(f"Failed to save downloaded classes file to {self.class_path}: {e}")
             return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during classes download: {e}")
            return False

    def _load_classes(self):
        """Loads class names from the specific local file path."""
        # Handle case where class_path might be None (e.g., custom model path used)
        if not self.class_path:
            logger.warning("Class file path is not set. Using default index class names.")
            return self._get_default_class_names()

        try:
            with open(self.class_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            if not classes:
                 logger.warning(f"Class file {self.class_path} was found but is empty. Using default index class names.")
                 return self._get_default_class_names()
            logger.debug(f"Loaded {len(classes)} classes from {self.class_path}")
            return classes
        except FileNotFoundError:
            logger.warning(f"Class file not found at {self.class_path}. Using default index class names.")
            return self._get_default_class_names()
        except Exception as e:
            logger.error(f"Failed to load or parse classes from {self.class_path}: {e}")
            return self._get_default_class_names()

    def _get_default_class_names(self):
        """Generates default class names (indices) based on model output or fallback."""
        try:
            # Try to get num_classes from model output shape if session is loaded
            if self.ort_session:
                num_outputs = self.ort_session.get_outputs()[0].shape[-1]
                if isinstance(num_outputs, int) and num_outputs > 0:
                    logger.info(f"Model expects {num_outputs} outputs. Generating indices as class names.")
                    return [str(i) for i in range(num_outputs)]
        except Exception as e:
            logger.warning(f"Could not determine expected number of classes from model session: {e}")

        logger.warning("Defaulting to 1000 indices as class names.")
        return [str(i) for i in range(1000)]  # Fallback default

    def _get_transform(self):
        """Defines the image preprocessing pipeline."""
        # Define ImageNet normalization parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize((self.CLASSIFIER_IMAGE_SIZE, self.CLASSIFIER_IMAGE_SIZE)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean, std)  # Normalize image
        ])

    def predict(self, image_path, top_k=5):
        """
        Performs inference on a single image loaded from disk.

        Args:
            image_path (str): Path to the image file.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple: (top_k_indices, top_k_confidences, top1_class_name, top1_confidence)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image from disk and delegate to predict_from_image
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image(self, image, top_k=5):
        """
        Performs inference on a single image provided as a PIL Image or numpy array.

        Args:
            image (PIL.Image or np.ndarray): The input image.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple: (top_k_indices, top_k_confidences, top1_class_name, top1_confidence)
        """
        # If image is a numpy array, convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Convert to numpy for ONNX Runtime
        ort_input = {self.ort_session.get_inputs()[0].name: input_tensor.numpy()}

        # Run inference
        ort_outs = self.ort_session.run(None, ort_input)
        logits = ort_outs[0]

        # Compute softmax probabilities (numerically stable)
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)

        # Get top-k predictions
        top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]
        top_k_confidences = probabilities[0][top_k_indices]

        # Get top-1 prediction and confidence
        top1_index = top_k_indices[0]
        top1_confidence = top_k_confidences[0]
        top1_class_name = self.classes[top1_index]

        return top_k_indices, top_k_confidences, top1_class_name, top1_confidence

