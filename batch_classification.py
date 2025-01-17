# ------------------------------------------------------------------------------
# Batch Image Classification Script
# ------------------------------------------------------------------------------

import os
import csv
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# ------------------------------------------------------------------------------
# Load Labels for the Classification Model
# ------------------------------------------------------------------------------

with open("models/ImageNetLabels.txt", "r") as f:
    IMAGENET_LABELS = [line.strip() for line in f]

# ------------------------------------------------------------------------------
# Load the Classification Model from TensorFlow Hub
# ------------------------------------------------------------------------------
#    https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
print("Loading classification model from TensorFlow Hub...")
model = hub.load(MODEL_URL)
print("Model loaded!")

# ------------------------------------------------------------------------------
# IMAGE CLASSIFICATION AND RETURN TOP-5
# ------------------------------------------------------------------------------

def classify_image(cropped_bgr):
    """
    Classifies an image using the loaded TensorFlow Hub model.

    :param cropped_bgr: Cropped image in BGR format (OpenCV format).
    :return: A list of tuples containing the top-5 predicted labels and their probabilities.
    """
    # Convert BGR -> RGB
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    # Resize to the model’s expected size
    resized = cv2.resize(cropped_rgb, (224, 224))

    # >>>> CHANGE #1: Convert to float32 and scale <<<<
    resized = np.array(resized, dtype=np.float32) / 255.0

    # Make a batch dimension: (1, 224, 224, 3)
    input_tensor = tf.convert_to_tensor([resized], dtype=tf.float32)

    # >>>> CHANGE #2: Just call the model with the tensor <<<<
    outputs = model(input_tensor)  # shape: (1, 1001) for typical ImageNet classification

    probs = outputs[0].numpy()  # shape: (1001,)

    # Use argsort to get top-5 indices
    top5_indices = probs.argsort()[-5:][::-1]  # last 5, reversed
    top5 = []
    for idx in top5_indices:
        label_str = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"Class_{idx}"
        prob_val = probs[idx]
        top5.append((label_str, float(prob_val)))
    return top5


# ------------------------------------------------------------------------------
# Batch Processing: Read and Classify Cropped Images from a CSV
# ------------------------------------------------------------------------------
CSV_INPUT = "output/all_bounding_boxes.csv"
CSV_OUTPUT = "output/all_bounding_boxes_classified.csv"

# We'll create a new CSV with extra columns: top5_label_1, top5_prob_1, ... top5_label_5, top5_prob_5
# Or you can store them in a single column—choose what you prefer.

with open(CSV_INPUT, mode="r", newline="") as fin, \
     open(CSV_OUTPUT, mode="w", newline="") as fout:

    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames + [
        "top1_label", "top1_prob",
        "top2_label", "top2_prob",
        "top3_label", "top3_prob",
        "top4_label", "top4_prob",
        "top5_label", "top5_prob"
    ]

    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Convert bounding box coords from string to int
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        # The "filename" column: we assume it’s a path like "frame_annotated_20230801_121314.jpg"
        image_path = os.path.join("output", row["filename"])  # or adapt to your own path structure

        if not os.path.exists(image_path):
            print(f"WARNING: Image file not found: {image_path}")
            # Write the row unchanged (with empty classification columns)
            writer.writerow(row)
            continue

        # 4.1) LOAD THE IMAGE
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Could not open image: {image_path}")
            writer.writerow(row)
            continue

        # 4.2) CROP
        # Ensure your bounding box is within the image
        h, w, _ = image_bgr.shape
        # Clip coords
        x1_clamped = max(0, min(x1, w-1))
        x2_clamped = max(0, min(x2, w-1))
        y1_clamped = max(0, min(y1, h-1))
        y2_clamped = max(0, min(y2, h-1))

        cropped_bgr = image_bgr[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        if cropped_bgr.size == 0:
            print(f"Empty crop at row: {row}")
            writer.writerow(row)
            continue

        # 4.3) CLASSIFY
        top5 = classify_image(cropped_bgr)  # list of (label, prob)

        # 4.4) Add the top-5 results to the CSV row
        # e.g. row["top1_label"] = top5[0][0], row["top1_prob"] = top5[0][1], etc.
        for i in range(5):
            label_i = top5[i][0]  # string
            prob_i = top5[i][1]  # float
            row[f"top{i+1}_label"] = label_i
            row[f"top{i+1}_prob"]  = f"{prob_i:.4f}"

        # Write updated row
        writer.writerow(row)

print(f"Done! Classified crops saved to {CSV_OUTPUT}")


