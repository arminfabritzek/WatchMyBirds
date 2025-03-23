import os
os.environ["HF_HOME"] = os.path.abspath("models")
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import numpy as np  # Import numpy


class ImageClassifier:
    def __init__(self, model_path, class_path="models/imagenet_classes.txt"):
        """
        Initializes the ImageClassifier.

        Args:
            model_path (str): Path to the ONNX model file.
            class_path (str, optional): Path to the file containing class names.
                                        Defaults to "imagenet_classes.txt".  Handles
                                        cases where the file might not exist.
        """
        self.model_path = model_path
        self.class_path = class_path

        # If the ONNX model file does not exist, export it from the pretrained model
        if not os.path.exists(self.model_path):
            print(f"ONNX model not found at {self.model_path}. Exporting pretrained EfficientNet-B0 to ONNX...")
            ImageClassifier.export_model(self.model_path)
        # If the class file does not exist, download imagenet_classes.txt
        if not os.path.exists(self.class_path):
            print(f"Class file not found at {self.class_path}. Downloading imagenet_classes.txt...")
            ImageClassifier.download_imagenet_classes(self.class_path)

        self.ort_session = ort.InferenceSession(self.model_path)
        self.classes = self._load_classes()
        self.transform = self._get_transform()

    @staticmethod
    def export_model(onnx_model_path):
        """Exports a pretrained EfficientNet-B0 model to ONNX format."""
        import torch
        import timm

        device = torch.device("cpu")
        # Load the pretrained EfficientNet-B0 model (simulate training)
        model = timm.create_model('efficientnet_b0', pretrained=True)
        model.eval()  # set model to evaluation mode

        # Create a dummy input with the expected shape
        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        # Export the model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"Model exported to ONNX format at: {onnx_model_path}")

    @staticmethod
    def download_imagenet_classes(dest_path):
        """Downloads the imagenet_classes.txt file from PyTorch Hub if needed."""
        import requests
        import hashlib
        import os

        url = "https://raw.githubusercontent.com/pytorch/hub/c7895df70c7767403e36f82786d6b611b7984557/imagenet_classes.txt"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download imagenet_classes.txt. Status code: {response.status_code}")
            return

        downloaded_data = response.content
        downloaded_hash = hashlib.sha256(downloaded_data).hexdigest()
        print(f"Downloaded imagenet_classes.txt hash: {downloaded_hash}")

        if os.path.exists(dest_path):
            with open(dest_path, "rb") as f:
                local_data = f.read()
            local_hash = hashlib.sha256(local_data).hexdigest()
            print(f"Local imagenet_classes.txt hash: {local_hash}")
        else:
            local_hash = None

        if local_hash != downloaded_hash:
            with open(dest_path, "wb") as f:
                f.write(downloaded_data)
            print(f"imagenet_classes.txt updated at: {dest_path}")
        else:
            print("Local imagenet_classes.txt is already up-to-date.")

    def _load_classes(self):
        """Loads class names from the specified file."""
        try:
            with open(self.class_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: Class file not found at {self.class_path}. Using index as class name.")
            return [str(i) for i in range(1000)]  # Return indices as strings

    def _get_transform(self):
        """Defines the image preprocessing pipeline."""
        # ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
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

