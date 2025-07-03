# ------------------------------------------------------------------------------
# web_interface.py
# ------------------------------------------------------------------------------

import os
import json
import math
from urllib.parse import parse_qs
import re
import logging
import csv
from flask import Flask, send_from_directory, Response
from dash import Dash, html, dcc, callback_context, ALL, Input, Output, State, no_update, ClientsideFunction
import dash_bootstrap_components as dbc
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import plotly.express as px
from config import load_config
config = load_config()

import zipfile
import io  # Create in-memory zip buffer
import base64  # Send zip data to dcc.Download
from dash.exceptions import PreventUpdate
import pandas as pd


# >>> Caching settings for gallery functions >>>
_CACHE_TIMEOUT = 10  # Set cache timeout in seconds
_cached_images = {
    "images": None,
    "timestamp": 0
}

def create_web_interface(detection_manager):
    """
    Creates and returns a web interface (Dash app and Flask server) for the project.
    Expects the following keys in params:
      - output_dir: Directory where detection images are saved.
      - video_capture: Object with a method generate_frames(width, fps).
      - output_resize_width: The width to which video frames should be resized.
      - STREAM_FPS: Frame rate for the video stream.
      - IMAGE_WIDTH: Width for thumbnail images.
      - RECENT_IMAGES_COUNT: Number of recent images to display.
      - PAGE_SIZE: Number of images per gallery page.
    """
    logger = logging.getLogger(__name__)

    output_dir = config["OUTPUT_DIR"]
    output_resize_width = config["STREAM_WIDTH_OUTPUT_RESIZE"]
    STREAM_FPS = config["STREAM_FPS"]
    CONFIDENCE_THRESHOLD_DETECTION = config["CONFIDENCE_THRESHOLD_DETECTION"]
    CLASSIFIER_CONFIDENCE_THRESHOLD = config["CLASSIFIER_CONFIDENCE_THRESHOLD"]
    FUSION_ALPHA = config["FUSION_ALPHA"]
    EDIT_PASSWORD = config["EDIT_PASSWORD"]
    logger.info(f"Loaded EDIT_PASSWORD: {'***' if EDIT_PASSWORD and EDIT_PASSWORD != 'default_pass' else '<Not Set or Default>'}")

    if EDIT_PASSWORD == "default_pass":
        logger.warning("EDIT_PASSWORD not set in .env file, using default. THIS IS INSECURE.")

    RECENT_IMAGES_COUNT = 10
    IMAGE_WIDTH = 150
    PAGE_SIZE = 50


    common_names_file = os.path.join(os.getcwd(), "assets", "common_names_DE.json")
    try:
        with open(common_names_file, "r", encoding="utf-8") as f:
            COMMON_NAMES = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load common names from {common_names_file}: {e}")
        COMMON_NAMES = {
            "Cyanistes_caeruleus": "Eurasian blue tit"
        }

    # >>> Helper Functions for CSV and File Operations >>>
    def get_csv_path(date_str_iso):
        """Gets the expected path to the images.csv file for a given date."""
        date_folder = date_str_iso.replace('-', '')  # Convert YYYY-MM-DD to YYYYMMDD
        return os.path.join(output_dir, date_folder, "images.csv")

    def read_csv_for_date(date_str_iso):
        """Reads the CSV for a specific date into a pandas DataFrame."""
        csv_path = get_csv_path(date_str_iso)
        if not os.path.exists(csv_path):
            return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
        try:
            # Specify dtype={'downloaded_timestamp': str} to avoid pandas interpreting it as date/time
            # Keep other columns as default or specify if needed
            return pd.read_csv(csv_path, keep_default_na=False, dtype={'downloaded_timestamp': str}).sort_values(by='timestamp', ascending=False)
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return pd.DataFrame()  # Return empty on error

    def write_csv_for_date(date_str_iso, df):
        """Writes a pandas DataFrame back to the CSV for a specific date."""
        csv_path = get_csv_path(date_str_iso)
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # Work on a copy to avoid SettingWithCopyWarning
            df = df.copy()

            # Ensure 'downloaded_timestamp' exists before writing; fill with empty string if not
            if 'downloaded_timestamp' not in df.columns:
                df.loc[:, 'downloaded_timestamp'] = ''
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)  # Use minimal quoting

            # Clear the cache as CSV has changed
            _cached_images["images"] = None
            _cached_images["timestamp"] = 0
            logger.info(f"Successfully wrote CSV for {date_str_iso}")
            return True
        except Exception as e:
            logger.error(f"Error writing CSV {csv_path}: {e}")
            return False

    def delete_image_files(relative_optimized_path):
        """Deletes original, optimized, and zoomed versions of an image."""
        base_path = os.path.join(output_dir, relative_optimized_path)
        original_path = base_path.replace("_optimized", "_original")
        zoomed_path = derive_zoomed_filename(base_path)  # Use existing helper

        deleted_count = 0
        for img_path in [original_path, base_path, zoomed_path]:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info(f"Deleted image file: {img_path}")
                    deleted_count += 1
            except OSError as e:
                logger.error(f"Error deleting file {img_path}: {e}")
        return deleted_count > 0  # Return True if at least one file was deleted

    def get_all_images():
        """
        Reads all per-day CSV files (from folders named YYYYMMDD) and returns a list of tuples:
          (timestamp, optimized image relative path, best_class, best_class_conf, top1_class_name, top1_confidence)
        Sorted by timestamp (newest first).
        """
        images = []
        # List subfolders in output_dir that match a day folder (e.g. "20250318")
        for item in os.listdir(output_dir):
            subfolder = os.path.join(output_dir, item)
            if os.path.isdir(subfolder) and re.match(r'\d{8}', item):
                csv_file = os.path.join(subfolder, "images.csv")
                if os.path.exists(csv_file):
                    try:
                        with open(csv_file, newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                # Extract values by column names.
                                timestamp = row.get("timestamp", "").strip()
                                optimized_name = row.get("optimized_name", "").strip()
                                best_class = row.get("best_class", "").strip()
                                best_class_conf = row.get("best_class_conf", "").strip()
                                top1_class = row.get("top1_class_name", "").strip()
                                top1_conf = row.get("top1_confidence", "").strip()
                                if not timestamp or not optimized_name:
                                    continue
                                # Construct relative path: folder/optimized_name
                                rel_path = os.path.join(item, optimized_name)
                                images.append((timestamp, rel_path, best_class, best_class_conf, top1_class, top1_conf))
                    except Exception as file_err:
                        logger.error(f"Error reading CSV file {csv_file}: {file_err}")
                        continue
        # Sort images by timestamp descending (YYYYMMDD_HHMMSS lexical order)
        images.sort(key=lambda x: x[0], reverse=True)
        return images

    def get_captured_images():
        """
        Returns a list of captured optimized images using the CSV-based approach.
        Uses caching to avoid repeated disk reads.
        """
        now = time.time()
        if _cached_images["images"] is not None and (now - _cached_images["timestamp"]) < _CACHE_TIMEOUT:
            return _cached_images["images"]
        images = get_all_images()
        _cached_images["images"] = images
        _cached_images["timestamp"] = now
        return images

    def get_captured_images_by_date():
        """
        Returns a dictionary grouping images by date (YYYY-MM-DD) using the CSV-based image list.
        The grouping is done by extracting the date from the optimized filename (which is expected to start with YYYYMMDD).
        """
        images = get_captured_images()  # now each element is a tuple: (timestamp, filename, best_class, best_class_conf, top1_class_name, top1_confidence)
        images_by_date = {}
        for timestamp, filename, best_class, best_class_conf, top1_class, top1_conf in images:
            base = os.path.basename(filename)
            match = re.match(r"(\d{8})_\d{6}.*\.jpg", base)
            if match:
                date_str = match.group(1)  # Extract YYYYMMDD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                if formatted_date not in images_by_date:
                    images_by_date[formatted_date] = []
                images_by_date[formatted_date].append((filename, best_class, best_class_conf, top1_class, top1_conf))
        return images_by_date

    def derive_zoomed_filename(optimized_filename: str,
                               optimized_suffix: str = "_optimized",
                               zoomed_suffix: str = "_zoomed") -> str:
        """Derives the zoomed image filename from the optimized filename."""
        return optimized_filename.replace(optimized_suffix, zoomed_suffix)

    def create_thumbnail_button(image_filename: str, index: int, id_type: str):
        """Creates a clickable thumbnail button with standard styling."""
        zoomed_filename = derive_zoomed_filename(image_filename)
        return html.Button(
            html.Img(
                src=f"/images/{zoomed_filename}",
                alt=f"Thumbnail of {zoomed_filename}",
                className="thumbnail-image",
                style={"width": f"{IMAGE_WIDTH}px"}
            ),
            id={'type': id_type, 'index': index},
            n_clicks=0,
            className="thumbnail-button" # Apply class to button
        )

    def create_image_modal_layout(image_filename: str, index: int, id_prefix: str):
        """Creates the modal layout content with standard styling."""
        original_filename = image_filename.replace("_optimized", "_original")
        pattern = r"(?:.*/)?(\d{8})_(\d{6})_([A-Za-z]+_[A-Za-z]+)_optimized\.jpg"
        match = re.match(pattern, image_filename)
        if match:
            date_str, time_str, class_name = match.groups()
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            formatted_class = class_name.replace('_', ' ')
            title_content = [f"{formatted_date} {formatted_time} - ", html.Em(formatted_class)]
        else:
            title_content = image_filename

        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(title_content), close_button=False),
                dbc.ModalBody(
                    html.Img(
                        src=f"/images/{image_filename}",
                        className="modal-image",
                        id={'type': f'{id_prefix}-modal-image', 'index': index}
                    )
                ),
                dbc.ModalFooter([
                    html.A(
                        dbc.Button("Download", color="secondary", target="_blank"),
                        href=f"/images/{original_filename}",
                        download=original_filename,
                        className="modal-download-link"
                    ),
                    dbc.Button(
                        "Close",
                        id={'type': f'{id_prefix}-close', 'index': index},
                        className="ms-auto",
                        n_clicks=0
                    )
                ]),
            ],
            id={'type': f'{id_prefix}-modal', 'index': index},
            is_open=False,
            size="lg",
        )

    def create_thumbnail_info_box(best_class, best_class_conf, top1_class, top1_conf):
        """
        Creates the standardized information box displayed below gallery thumbnails.

        Args:
            best_class (str): The scientific name from the detector.
            best_class_conf (str or float): The confidence score from the detector.
            top1_class (str): The scientific name from the classifier.
            top1_conf (str or float): The confidence score from the classifier.

        Returns:
            html.Div: The Dash HTML component for the info box.
        """
        # Basic error checking/default values for confidence if they might be missing/invalid
        try:
            detector_conf_percent = int(float(best_class_conf) * 100)
        except (ValueError, TypeError):
            detector_conf_percent = 0

        try:
            classifier_conf_percent = int(float(top1_conf) * 100)
        except (ValueError, TypeError):
            classifier_conf_percent = 0

        # Format names (handle potential None or empty strings if necessary)
        best_class_sci = best_class.replace('_', ' ') if best_class else "N/A"
        top1_class_sci = top1_class.replace('_', ' ') if top1_class else "N/A"  # We need this formatted for consistency later, even if not displayed directly in Classifier line

        common_name_display = COMMON_NAMES.get(best_class, best_class_sci) if best_class else "Unbekannt"

        return html.Div([
            # Line 1: Common Name (from best_class)
            html.Span(
                html.Strong(common_name_display),
                className="info-common-name"
            ),
            # Line 2: Scientific Name (from best_class)
            html.Span(
                ["(", html.I(best_class_sci), ")"],
                className="info-scientific-name"
            ),
            # Line 3: Classifier Confidence (ONLY confidence percentage)
            html.Span(
                f"Classifier: {classifier_conf_percent}%",
                className="info-classifier-conf"
            ),
            # Line 4: Detector Confidence (WITH formatted best_class name)
            html.Span(
                [
                    "Detector: ",
                    html.I(best_class_sci),  # Use formatted detector name
                    f" ({detector_conf_percent}%)"  # Detector confidence
                ],
                className="info-detector-conf"
            )
        ], className="thumbnail-info")

    def create_fused_agreement_info_box(final_class, combined_score, conf_d, conf_c):
        """Creates an info box for the agreement-based fusion summary."""
        final_class_sci = final_class.replace('_', ' ') if final_class else "N/A"
        common_name_display = COMMON_NAMES.get(final_class, final_class_sci) if final_class else "Unbekannt"
        combined_score_percent = int(combined_score * 100)  # Product score needs scaling
        detector_conf_percent = int(conf_d * 100)
        classifier_conf_percent = int(conf_c * 100)

        return html.Div([
            html.Span(html.Strong(common_name_display), className="info-common-name"),
            html.Span(["(", html.I(final_class_sci), ")"], className="info-scientific-name"),
            # Show the combined score prominently
            html.Span(f"Produkt: {combined_score_percent}%", className="info-combined-conf"),
            # Optionally show original scores for context (smaller font?)
            html.Span(f"(Det: {detector_conf_percent}%, Cls: {classifier_conf_percent}%)", className="info-detector-conf")
        ], className="thumbnail-info")

    def create_fused_weighted_info_box(final_class, combined_score, best_class, conf_d, conf_c):
        """Creates an info box for the weighted fusion summary."""
        final_class_sci = final_class.replace('_', ' ') if final_class else "N/A"
        common_name_display = COMMON_NAMES.get(final_class, final_class_sci) if final_class else "Unbekannt"
        combined_score_percent = int(combined_score * 100)  # Weighted score already 0-1
        detector_conf_percent = int(conf_d * 100)
        classifier_conf_percent = int(conf_c * 100)
        detector_class_sci = best_class.replace('_', ' ') if best_class else "N/A"

        # Indicate if detector disagreed
        disagreement_note = ""
        if final_class != best_class and best_class:
             disagreement_note = f" (Det: {detector_class_sci})"  # Show what detector thought

        return html.Div([
            html.Span(html.Strong(common_name_display), className="info-common-name"),
            html.Span(["(", html.I(final_class_sci), ")", disagreement_note], className="info-scientific-name"),
            # Show the combined score prominently
            html.Span(f"Gewichtet: {combined_score_percent}%", className="info-combined-conf"),
            html.Span(f"(Det: {detector_conf_percent}%, Cls: {classifier_conf_percent}%)", className="info-regular")
        ], className="thumbnail-info")

    def create_subgallery_modal(image_filename: str, index: int):
        return create_image_modal_layout(image_filename, index, 'subgallery-modal')

    def generate_daily_fused_summary_agreement(date_str_iso, images_for_date):
        """Generates a gallery based on agreement and multiplicative score for a specific date."""
        best_results_per_species = {}  # key: species_name, value: (combined_score, path, best_class, conf_d, top1_class, conf_c)
        # Input images_for_date is list of: (path, best_class, best_class_conf, top1_class, top1_conf)

        for path, best_class, best_class_conf, top1_class, top1_conf in images_for_date:
            if not best_class or not top1_class: continue  # Need both classes

            try:
                conf_d = float(best_class_conf)
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidences aren't valid numbers

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION and
                conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD and
                best_class == top1_class
            )

            if is_valid:
                final_class = best_class
                combined_score = conf_d * conf_c  # Multiplicative score

                # Check if this is the best score found so far for this species ON THIS DAY
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[0]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (combined_score, path, best_class, conf_d, top1_class, conf_c)

        # Convert dict to list for sorting/display
        summary_data = [
            (species, data[0], data[1], data[3], data[5]) # (final_class, combined_score, path, conf_d, conf_c)
            for species, data in best_results_per_species.items()
        ]

        # Sort by combined score descending for daily view
        summary_data.sort(key=lambda x: x[1], reverse=True)
        summary_data = summary_data[:RECENT_IMAGES_COUNT] # Limit count for daily view

        # --- Build Gallery ---
        gallery_items = []
        modals = []  # Keep modals separate

        if not summary_data:
            # Return placeholder component
            return html.P(f"Keine übereinstimmenden Erkennungen für {date_str_iso} gefunden.", className="text-center text-muted small")

        base_index_str = date_str_iso.replace('-', '')
        thumbnail_id_type = 'daily-fused-agreement-thumbnail'
        modal_id_prefix = 'daily-fused-agreement'

        for i, (final_class, combined_score, img, conf_d, conf_c) in enumerate(summary_data):
            unique_index = f"{base_index_str}-agree-{i}"
            tile = html.Div([
                create_thumbnail_button(img, unique_index, thumbnail_id_type),
                create_fused_agreement_info_box(final_class, combined_score, conf_d, conf_c)
            ], className="gallery-tile")
            gallery_items.append(tile)
            # Create and add modal to the separate list
            modals.append(create_image_modal_layout(img, unique_index, modal_id_prefix))

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_daily_fused_summary_weighted(date_str_iso, images_for_date):
        """Generates a gallery based on weighted score for a specific date."""
        best_results_per_species = {}  # key: species_name (top1_class), value: (combined_score, path, best_class, conf_d, top1_class, conf_c)
        # Input images_for_date is list of: (path, best_class, best_class_conf, top1_class, top1_conf)

        for path, best_class, best_class_conf, top1_class, top1_conf in images_for_date:
            if not top1_class: continue # Require a classifier result

            try:
                conf_d = float(best_class_conf) if best_class_conf else 0.0
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION and
                conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
            )

            if is_valid:
                final_class = top1_class  # Prioritize classifier label
                combined_score = FUSION_ALPHA * conf_d + (1.0 - FUSION_ALPHA) * conf_c

                # Check if this is the best score found so far for this final_class ON THIS DAY
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[0]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (combined_score, path, best_class, conf_d, top1_class, conf_c)

        # Convert dict to list for sorting/display
        summary_data = [
             # (final_class, combined_score, path, best_class, conf_d, conf_c)
            (species, data[0], data[1], data[2], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort by combined score descending for daily view
        summary_data.sort(key=lambda x: x[1], reverse=True)
        summary_data = summary_data[:RECENT_IMAGES_COUNT]  # Limit count for daily view

        # --- Build Gallery ---
        gallery_items = []
        modals = []  # Keep modals separate

        if not summary_data:
            # Return placeholder and empty list for modals
            return html.P(f"Keine gewichteten Erkennungen für {date_str_iso} gefunden.", className="text-center text-muted small")

        base_index_str = date_str_iso.replace('-', '')
        thumbnail_id_type = 'daily-fused-weighted-thumbnail'
        modal_id_prefix = 'daily-fused-weighted'

        for i, (final_class, combined_score, img, best_class_orig, conf_d_orig, conf_c_orig) in enumerate(summary_data):
            unique_index = f"{base_index_str}-weight-{i}"
            tile = html.Div([
                create_thumbnail_button(img, unique_index, thumbnail_id_type),
                create_fused_weighted_info_box(final_class, combined_score, best_class_orig, conf_d_orig, conf_c_orig)
            ], className="gallery-tile")
            gallery_items.append(tile)
            # Create and add modal to the separate list
            modals.append(create_image_modal_layout(img, unique_index, modal_id_prefix))

        # Return the Div containing only gallery items, and the list of modals
        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_all_time_detector_summary():
        """
        Generates a gallery of the best detector result for each species across all time,
        considering only detections above the confidence threshold.
        """
        all_images = get_captured_images()  # Get all images (uses cache)
        best_images_all_time = {}  # Dictionary to store best image per species

        # Find the highest confidence detection for each species THAT MEETS THE THRESHOLD
        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not best_class: continue  # Skip if detector class is missing
            try:
                conf_val = float(best_class_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidence is not a valid number

            if conf_val < CONFIDENCE_THRESHOLD_DETECTION:
                continue  # Skip if detection confidence is below the threshold

            # Get the current best confidence stored for this species (default to -1.0 if not found)
            current_best_conf = best_images_all_time.get(best_class, (None, -1.0, None, None))[1]

            # If the current image's confidence is higher than the stored one, update it
            if conf_val > current_best_conf:
                best_images_all_time[best_class] = (path, conf_val, top1_class, top1_conf)

        # Create the list of unique detections from the dictionary
        all_time_unique_detector = [
            (data[0], species, data[1], data[2], data[3])  # (path, best_class, conf_val, top1_class, top1_conf)
            for species, data in best_images_all_time.items()
        ]
        # Sort alphabetically by the common name of the detected species (best_class)
        all_time_unique_detector.sort(key=lambda x: COMMON_NAMES.get(x[1], x[1]))

        gallery_items, modals = [], []
        if not all_time_unique_detector:
            return html.P(
                f"Keine Arten (Detector) über Schwellenwert ({int(CONFIDENCE_THRESHOLD_DETECTION * 100)}%) bisher erfasst.",
                className="text-center text-muted"
            )

        thumbnail_id_type = 'alltime-detector-thumbnail'
        modal_id_prefix = 'alltime-detector'

        # Generate gallery tiles and modals
        for i, (img, best_class, confidence, top1_class, top1_conf) in enumerate(all_time_unique_detector):
            tile = html.Div([
                create_thumbnail_button(img, i, thumbnail_id_type),
                create_thumbnail_info_box(best_class, confidence, top1_class, top1_conf)  # Standard info box
            ], className="gallery-tile")
            gallery_items.append(tile)
            modals.append(create_image_modal_layout(img, i, modal_id_prefix))

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_all_time_classifier_summary():
        """Generates a gallery of the best classifier result for each species across all time."""
        all_images = get_captured_images()  # Get all images (uses cache)
        best_classifier_images_all_time = {}  # Dictionary to store best image per classified species

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not top1_class: continue  # Skip if classifier class is missing
            try: conf_val = float(top1_conf)
            except (ValueError, TypeError): continue

            if conf_val < CLASSIFIER_CONFIDENCE_THRESHOLD: continue

            current_best_conf = best_classifier_images_all_time.get(top1_class, (None, None, None, None, -1.0))[4]
            if conf_val > current_best_conf:
                # Store original detector info too
                best_classifier_images_all_time[top1_class] = (path, best_class, best_class_conf, top1_class, conf_val)

        all_time_unique_classifier = [ (data[0], data[1], data[2], species, data[4]) for species, data in best_classifier_images_all_time.items() ]
        # Sort by common name of the CLASSIFIED species (top1_class)
        all_time_unique_classifier.sort(key=lambda x: COMMON_NAMES.get(x[3], x[3]))

        gallery_items, modals = [], []
        if not all_time_unique_classifier: return html.P(f"Keine Arten (Classifier) bisher erfasst (Schwelle: {int(CLASSIFIER_CONFIDENCE_THRESHOLD * 100)}%).", className="text-center text-muted")

        thumbnail_id_type = 'alltime-classifier-thumbnail'
        modal_id_prefix = 'alltime-classifier'

        for i, (img, best_class, best_class_conf, top1_class, top1_conf) in enumerate(all_time_unique_classifier):
             tile = html.Div([
                 create_thumbnail_button(img, i, thumbnail_id_type),
                 create_thumbnail_info_box(best_class, best_class_conf, top1_class, top1_conf) # Standard info box
             ], className="gallery-tile")
             gallery_items.append(tile)
             modals.append(create_image_modal_layout(img, i, modal_id_prefix))

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_all_time_fused_summary_agreement():
        """Generates a gallery based on agreement and multiplicative score."""
        all_images = get_captured_images()
        best_results_per_species = {}  # key: species_name, value: (combined_score, path, best_class, conf_d, top1_class, conf_c)

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not best_class or not top1_class: continue  # Need both classes

            try:
                conf_d = float(best_class_conf)
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidences aren't valid numbers

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION and
                conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD and
                best_class == top1_class
            )

            if is_valid:
                final_class = best_class
                combined_score = conf_d * conf_c  # Multiplicative score

                # Check if this is the best score found so far for this species
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[0]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (combined_score, path, best_class, conf_d, top1_class, conf_c)

        # Convert dict to list for sorting/display
        # List items: (final_class, combined_score, path, conf_d, conf_c) <= simplified for info box
        summary_data = [
            (species, data[0], data[1], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort alphabetically by common name of the agreed class
        summary_data.sort(key=lambda x: COMMON_NAMES.get(x[0], x[0]))

        # --- Build Gallery ---
        gallery_items = []
        modals = []
        if not summary_data:
             return html.P("Keine übereinstimmenden Erkennungen über Schwellenwerten gefunden.", className="text-center text-muted")

        thumbnail_id_type = 'alltime-fused-agreement-thumbnail'
        modal_id_prefix = 'alltime-fused-agreement'

        for i, (final_class, combined_score, img, conf_d, conf_c) in enumerate(summary_data):
             tile = html.Div([
                 create_thumbnail_button(img, i, thumbnail_id_type),
                 create_fused_agreement_info_box(final_class, combined_score, conf_d, conf_c)  # Use specific info box
             ], className="gallery-tile")
             gallery_items.append(tile)
             modals.append(create_image_modal_layout(img, i, modal_id_prefix))  # Standard modal layout

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_all_time_fused_summary_weighted():
        """Generates a gallery based on weighted score, prioritizing classifier label."""
        all_images = get_captured_images()
        best_results_per_species = {}  # key: species_name (top1_class), value: (combined_score, path, best_class, conf_d, top1_class, conf_c)

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            # Require a classifier result for this method
            if not top1_class: continue

            try:
                conf_d = float(best_class_conf) if best_class_conf else 0.0 # Default detector conf to 0 if missing
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if classifier conf is invalid

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION and
                conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
            )

            if is_valid:
                final_class = top1_class  # Prioritize classifier label
                combined_score = FUSION_ALPHA * conf_d + (1.0 - FUSION_ALPHA) * conf_c  # Weighted score

                # Check if this is the best score found so far for this final_class (top1_class)
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[0]
                if combined_score > current_best_score:
                     # Store original detector info as well
                    best_results_per_species[final_class] = (combined_score, path, best_class, conf_d, top1_class, conf_c)

        # Convert dict to list for sorting/display
        # List items: (final_class, combined_score, path, best_class, conf_d, conf_c) <= Need original detector class for info box
        summary_data = [
            (species, data[0], data[1], data[2], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort alphabetically by common name of the FINAL class (top1_class)
        summary_data.sort(key=lambda x: COMMON_NAMES.get(x[0], x[0]))

        # --- Build Gallery ---
        gallery_items = []
        modals = []
        if not summary_data:
             return html.P("Keine Erkennungen über Schwellenwerten für gewichtete Bewertung gefunden.", className="text-center text-muted")

        thumbnail_id_type = 'alltime-fused-weighted-thumbnail'
        modal_id_prefix = 'alltime-fused-weighted'

        for i, (final_class, combined_score, img, best_class, conf_d, conf_c) in enumerate(summary_data):
             tile = html.Div([
                 create_thumbnail_button(img, i, thumbnail_id_type),
                 create_fused_weighted_info_box(final_class, combined_score, best_class, conf_d, conf_c)  # Use specific info box
             ], className="gallery-tile")
             gallery_items.append(tile)
             modals.append(create_image_modal_layout(img, i, modal_id_prefix))  # Standard modal layout

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def species_summary_layout():
        """Layout for the all-time species summary page."""
        detector_summary = dcc.Loading(type="circle", children=generate_all_time_detector_summary())
        classifier_summary = dcc.Loading(type="circle", children=generate_all_time_classifier_summary())
        fused_agreement_summary = dcc.Loading(type="circle", children=generate_all_time_fused_summary_agreement())
        fused_weighted_summary = dcc.Loading(type="circle", children=generate_all_time_fused_summary_weighted())

        return dbc.Container([
            generate_navbar(),  # Include navbar
            html.H1("Artenübersicht (Alle Tage)", className="text-center my-3"),

            # --- Section 1: Best Detector ---
            dbc.Row([
                dbc.Col([
                    html.H3("Beste Detektion pro Art", className="text-center mt-4 mb-3"),
                    html.P(f"Zeigt das Bild mit der höchsten Detector-Konfidenz (>= {int(CONFIDENCE_THRESHOLD_DETECTION*100)}%) für jede klassifizierte Art.", className="text-center text-muted small mb-3"),
                    detector_summary
                ], width=12),
            ]),
            html.Hr(className="my-4"),

             # --- Section 2: Best Classifier ---
            dbc.Row([
                dbc.Col([
                    html.H3("Beste Klassifizierung pro Art", className="text-center mt-4 mb-3"),
                    html.P(f"Zeigt das Bild mit der höchsten Classifier-Konfidenz (>= {int(CLASSIFIER_CONFIDENCE_THRESHOLD*100)}%) für jede klassifizierte Art.", className="text-center text-muted small mb-3"),
                    classifier_summary
                ], width=12),
            ]),
            html.Hr(className="my-4"),

            # --- Section 3: Fused - Agreement & Multiplicative Score ---
            dbc.Row([
                dbc.Col([
                    html.H3("Agreement & Produkt-Score", className="text-center mt-4 mb-3"),
                     html.P(f"Zeigt das Bild mit dem höchsten Produkt-Score (Detektor x Classifier), nur wenn beide Modelle zustimmen und über ihren Schwellenwerten liegen (Det >= {int(CONFIDENCE_THRESHOLD_DETECTION*100)}%, Cls >= {int(CLASSIFIER_CONFIDENCE_THRESHOLD*100)}%).", className="text-center text-muted small mb-3"),
                    fused_agreement_summary
                ], width=12),
            ]),
            html.Hr(className="my-4"),

            # --- Section 4: Fused - Weighted Score ---
             dbc.Row([
                dbc.Col([
                    html.H3(f"Gewichteter Score, α={FUSION_ALPHA}", className="text-center mt-4 mb-3"),
                     html.P(f"Zeigt das Bild mit dem höchsten gewichteten Score ({FUSION_ALPHA*100:.0f}% Detektor + { (1-FUSION_ALPHA)*100:.0f}% Classifier), wenn beide Modelle über ihren Schwellenwerten liegen. Bei Uneinigkeit wird die Klasse des Classifiers verwendet.", className="text-center text-muted small mb-3"),
                    fused_weighted_summary
                ], width=12),
            ]),

        ], fluid=True)

    # -------------------------------------
    # Edit Page Layout Generation
    # -------------------------------------
    def generate_edit_page(date_str_iso):
        """Generates the layout for the image editing page (simplified)."""
        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return dbc.Container([
                generate_navbar(),
                html.H2(f"Edit Bilder vom {date_str_iso}", className="text-center my-3"),
                dbc.Alert(f"No images found or error reading data for {date_str_iso}.", color="warning"),
                dbc.Button("Back to Subgallery", href=f"/gallery/{date_str_iso}", color="secondary", className="me-2"),
                dbc.Button("Back to Main Gallery", href="/gallery", color="secondary"),
            ], fluid=True)

        image_tiles = []

        # Generate image tiles with checkboxes
        for df_index, row in df.iterrows():  # No need for enumerate index anymore
            relative_path = os.path.join(date_str_iso.replace('-', ''), row['optimized_name'])
            checklist_value = relative_path  # Use relative path as the unique identifier

            info_box = create_thumbnail_info_box(
                row.get('best_class', ''),
                row.get('best_class_conf', ''),
                row.get('top1_class_name', ''),
                row.get('top1_confidence', '')
            )
            downloaded_ts = row.get('downloaded_timestamp', '')
            if downloaded_ts and str(downloaded_ts).strip():
                info_box.children.append(
                    html.Span(f"Downloaded", className="info-download-status text-success small")
                )

            zoomed_filename = derive_zoomed_filename(relative_path)

            # --- Simplified Checkbox Placement ---
            checkbox_component = dbc.Checkbox(
                id={'type': 'edit-image-checkbox', 'index': checklist_value},
                value=False,
                className="edit-checkbox",
            )

            # Determine ClassName for the Tile ---
            tile_classname = "gallery-tile edit-tile"  # Base classes
            if downloaded_ts and str(downloaded_ts).strip():
                tile_classname += " downloaded-image"  # Add class if downloaded

            image_tile = html.Div([
                checkbox_component,
                html.Img(
                    src=f"/images/{zoomed_filename}",
                    alt=f"Thumbnail {row['optimized_name']}",
                    className="thumbnail-image",
                    style={"width": f"{IMAGE_WIDTH}px", 'cursor': 'pointer'}
                ),
                info_box
            ], className=tile_classname)  # Use the determined classname

            image_tiles.append(image_tile)

        return dbc.Container([
            generate_navbar(),
            html.H2(f"Edit Images for {date_str_iso}", className="text-center my-3"),
            # Navigation Buttons
            html.Div([
                dbc.Button("Back to Subgallery", href=f"/gallery/{date_str_iso}", color="secondary", outline=True,
                           className="me-2"),
                dbc.Button("Back to Main Gallery", href="/gallery", color="secondary", outline=True),
            ], className="mb-3"),

            # Action Buttons, Confirmation, Store, Download
            dbc.Row([
                dbc.Col(dbc.Button("Delete Selected Images", id="delete-button", color="danger", className="me-2"),
                        width="auto"),
                dbc.Col(dbc.Button("Download Selected Images", id="download-button", color="success"), width="auto"),
            ], justify="start", className="mb-3"),
            dcc.ConfirmDialog(
                id='confirm-delete',
                message=f'Are you sure you want to permanently delete the selected images and their CSV entries for {date_str_iso}? This cannot be undone.',
            ),
            dcc.Store(id='selected-images-store', data=[]),  # Store is now updated by Python callback
            dcc.Download(id="download-zip"),
            html.Div(id="edit-status-message"),

            # The Grid containing the tiles
            html.Div(image_tiles, className="gallery-grid-container", id="edit-gallery-grid"),  # Keep ID for JS

            # Bottom Action Buttons
            dbc.Row([
                dbc.Col(
                    dbc.Button("Delete Selected Images", id="delete-button-bottom", color="danger", className="me-2"),
                    width="auto"),
                dbc.Col(dbc.Button("Download Selected Images", id="download-button-bottom", color="success"),
                        width="auto"),
            ], justify="start", className="mt-3"),

        ], fluid=True, id="edit-page-container")

    def generate_navbar():
        """Creates the navbar with the logo for gallery pages."""
        return dbc.NavbarSimple(
            brand=html.Img(
                src="/assets/WatchMyBirds.png",
                className="img-fluid round-logo",
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Galerie", href="/gallery", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Artenübersicht", href="/species", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center custom-navbar"
        )

    def generate_gallery():
        """Generates the main gallery page with daily subgallery links, including the logo."""
        images_by_date = get_captured_images_by_date()

        # Sort dates descending (newest first)
        sorted_dates = sorted(images_by_date.keys(), reverse=True)

        grid_items = []
        if not sorted_dates:
            grid_items.append(html.P("Bisher keine Bilder in der Galerie.", className="text-center text-muted mt-5"))
        else:
            for date in sorted_dates:
                images = images_by_date[date]
                if not images: continue  # Skip if a date folder exists but is empty
                # Use the first image of the day as the representative thumbnail
                # Ensure the tuple structure is correct (filename is the first element)
                thumbnail_rel_path = images[0][0]
                # Derive the zoomed filename for the thumbnail link display
                thumbnail_display_path = derive_zoomed_filename(thumbnail_rel_path)

                grid_items.append(
                    html.Div([
                        html.A(
                            html.Img(
                                src=f"/images/{thumbnail_display_path}",
                                className="main-gallery-image",
                                style={"width": f"{IMAGE_WIDTH}px"}
                            ),
                            href=f"/gallery/{date}"  # Link to subgallery
                        ),
                        html.P(date, className="main-gallery-date")
                    ], className="main-gallery-item")
                )

        content = html.Div(
            grid_items,
            className="gallery-grid-container"
        )
        return dbc.Container([
            generate_navbar(),
            html.H1("Galerie", className="text-center my-3"),
            content  # Add the container with grid items
        ], fluid=True)

    def generate_subgallery(date, page=1, include_header=True):
        """
        Generates the content for a specific date's subgallery page,
        including daily species summaries, pagination, loading indicator,
        and empty state handling.
        """
        # Get today's date for comparison >>> ---
        today_date_iso = datetime.now().strftime("%Y-%m-%d")
        df = read_csv_for_date(date)
        # get_captured_images_by_date() is no longer the primary source here,
        # but still used by summaries if page == 1 and include_header == True
        images_by_date_for_summaries = get_captured_images_by_date()
        images_for_summary = images_by_date_for_summaries.get(date, [])

        if df.empty and not images_for_summary:  # Check both in case one fails but not the other
             # Handle case where no data exists for the date
             return dbc.Container([
                 generate_navbar(),
                 html.H2(f"Bilder vom {date}", className="text-center"),
                 dbc.Alert(f"Keine Bilder für den {date} gefunden.", color="info"),
                 dbc.Button("Zurück zur Galerieübersicht", href="/gallery", color="secondary", outline=True)
             ], fluid=True)

        images_by_date = get_captured_images_by_date()
        images_for_this_date = images_by_date.get(date, [])
        total_images = len(images_for_this_date)
        total_pages = math.ceil(total_images / PAGE_SIZE) or 1
        page = max(1, min(page, total_pages))
        start_index = (page - 1) * PAGE_SIZE
        end_index = page * PAGE_SIZE
        # Slice the DataFrame for pagination
        page_df = df.iloc[start_index:end_index]
        page_images = images_for_this_date[start_index:end_index]

        # --- Pagination Controls ---
        page_links = []
        for p in range(1, total_pages + 1):
            if p == page:
                link = dbc.Button(str(p), color="primary", disabled=True, className="mx-1", size="sm")
            else:
                link = dbc.Button(str(p), color="secondary", href=f"/gallery/{date}?page={p}", className="mx-1",
                                  size="sm")
            page_links.append(link)

        # --- Define Top Controls ---
        pagination_controls_top = html.Div(
            page_links,  # Use the list of links
            className="pagination-controls pagination-top",
            id="pagination-top"  # Keep this ID for the scroll target
        )

        # --- Define Bottom Controls ---
        # Create a new Div, potentially cloning the links or just reusing the list
        pagination_controls_bottom = html.Div(
            page_links,  # Reuse the list of links
            className="pagination-controls pagination-bottom",
            # id="pagination-bottom" # Optional: Add a UNIQUE ID if needed, otherwise omit
        )

        # --- Main Paginated Gallery Items and Modals ---
        gallery_items = []
        subgallery_modals = []

        # Define ID types/prefixes needed within the loop
        button_id_type = 'subgallery-thumbnail'
        modal_id_prefix = 'subgallery-modal'

        # Iterate over the DataFrame slice >>> ---
        if page_df.empty:
            gallery_items.append(html.P(f"Keine Bilder für den {date} auf dieser Seite gefunden.",
                                        className="text-center text-muted mt-4 mb-4"))
        else:
            for i, row in page_df.iterrows():
                # Construct the relative path from date and optimized_name
                relative_path = os.path.join(date.replace('-', ''), row['optimized_name'])
                # Get other data from the row
                best_class = row.get('best_class', '')
                best_class_conf = row.get('best_class_conf', '')
                top1_class = row.get('top1_class_name', '')
                top1_conf = row.get('top1_confidence', '')
                downloaded_ts = row.get('downloaded_timestamp', '')  # Get download status

                unique_subgallery_index = f"{date.replace('-', '')}-sub-{start_index + i}"  # Adjust index calculation slightly if needed based on iloc behavior vs enumerate

                # Add class if downloaded
                tile_classname = "gallery-tile"
                if downloaded_ts and str(downloaded_ts).strip():  # Check if timestamp exists and is not empty/whitespace
                    tile_classname += " downloaded-image"

                info_box = create_thumbnail_info_box(best_class, best_class_conf, top1_class, top1_conf)
                # Add downloaded text to info box
                if downloaded_ts and str(downloaded_ts).strip():
                     info_box.children.append(
                         html.Span(f"Downloaded", className="info-download-status text-success small")
                     )

                tile = html.Div([
                    create_thumbnail_button(relative_path, unique_subgallery_index, button_id_type),
                    info_box  # Use the potentially modified info_box
                ], className=tile_classname)  # Use the potentially modified classname

                gallery_items.append(tile)
                # Use relative_path for creating modals as well
                subgallery_modals.append(create_subgallery_modal(relative_path, unique_subgallery_index))

        # --- Main Paginated Content Area with Loading ---
        gallery_grid = html.Div(gallery_items, className="gallery-grid-container")
        loading_wrapper = dcc.Loading(
            id=f"loading-subgallery-{date}-{page}",
            type="circle",
            children=gallery_grid
        )

        # --- Final Page Structure ---
        container_elements = []
        if include_header:
            container_elements.append(generate_navbar())
            # Group Back and Edit buttons
            header_buttons = [
                 dbc.Button("Zurück zur Galerieübersicht", href="/gallery", color="secondary", className="mb-3 mt-3 me-2", outline=True)
            ]
            # Conditionally add Edit Button
            if date != today_date_iso:
                 # Button triggers modal, ID includes the date
                 header_buttons.append(
                     dbc.Button(
                         "Diesen Tag bearbeiten",
                         id={'type': 'open-edit-modal-button', 'date': date}, # New ID pattern
                         color="warning", size="sm", className="mb-3 mt-3",
                         n_clicks=0
                    )
                 )
            else:
                 header_buttons.append(
                     dbc.Button("Bearbeiten (Nur vergangene Tage)", color="warning", size="sm", className="mb-3 mt-3", disabled=True)
                 )
            container_elements.append(html.Div(header_buttons))

            container_elements.append(html.H2(f"Bilder vom {date}", className="text-center"))
            container_elements.append(html.P(f"Seite {page} von {total_pages} ({total_images} Bilder insgesamt)", className="text-center text-muted small"))

        # --- Add Daily Summary Content ONLY if page is 1 ---
        if page == 1 and include_header:
            agreement_summary_content = generate_daily_fused_summary_agreement(date, images_for_this_date)
            weighted_summary_content = generate_daily_fused_summary_weighted(date, images_for_this_date)
            container_elements.extend([
                html.Hr(),
                html.H4("Tagesübersicht: Agreement & Produkt-Score", className="text-center mt-4"),
                agreement_summary_content,
                html.H4(f"Tagesübersicht: Gewichteter Score, α={FUSION_ALPHA}", className="text-center mt-4"),
                weighted_summary_content,
                html.Hr(),
            ])

        # Add Header for the main paginated gallery section
        # container_elements.append(html.H4("Alle Bilder", className="text-center mt-4"))

        # Add pagination and main loading wrapper
        container_elements.extend([
            pagination_controls_top,
            loading_wrapper,
            pagination_controls_bottom
        ])

        # Collect ALL Modals
        all_modals = subgallery_modals
        container_elements.extend(all_modals)

        return dbc.Container(container_elements, fluid=True)

    def generate_video_feed():
        # Load placeholder once
        static_placeholder_path = "assets/static_placeholder.jpg"
        if os.path.exists(static_placeholder_path):
            static_placeholder = cv2.imread(static_placeholder_path)
            if static_placeholder is not None:
                original_h, original_w = static_placeholder.shape[:2]
                ratio = original_h / float(original_w)
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * ratio)
                static_placeholder = cv2.resize(static_placeholder, (placeholder_w, placeholder_h))
            else:
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * 9 / 16)
                static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
        else:
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

        # Parameters for text overlay
        padding_x_percent = 0.005
        padding_y_percent = 0.04
        min_font_size = 12
        min_font_size_percent = 0.05

        while True:
            start_time = time.time()
            # Retrieve the most recent display frame (raw or optimized)
            frame = detection_manager.get_display_frame()
            if frame is not None:
                # Assume we want to resize using the original resolution from video capture.
                w, h = detection_manager.video_capture.resolution
                output_resize_height = int(h * output_resize_width / w)
                resized_frame = cv2.resize(frame, (output_resize_width, output_resize_height))
                # Overlay current timestamp
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            else:
                # If no frame, use the placeholder.
                pil_image = Image.fromarray(cv2.cvtColor(static_placeholder, cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            padding_x = int(img_width * padding_x_percent)
            padding_y = int(img_height * padding_y_percent)
            custom_font = ImageFont.load_default()
            timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
            bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img_width - text_width - padding_x
            text_y = img_height - text_height - padding_y
            draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")
            # Convert back to OpenCV BGR format
            frame_with_timestamp = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', frame_with_timestamp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            elapsed = time.time() - start_time
            desired_frame_time = 1.0 / STREAM_FPS
            if elapsed < desired_frame_time:
                time.sleep(desired_frame_time - elapsed)

    def generate_hourly_detection_plot():
        # Get all captured images
        all_images = get_captured_images()
        today_str = datetime.now().strftime("%Y%m%d")

        # Initialize count for all 24 hours
        counts = {f"{hour:02d}": 0 for hour in range(24)}

        # Count detections per hour for today
        for ts, _, _, _, _, _ in all_images:
            if ts.startswith(today_str):
                # Assuming timestamp format "YYYYMMDD_HHMMSS"
                hour = ts[9:11]
                if hour in counts:
                    counts[hour] += 1

        # Create lists for plotting
        hours = list(counts.keys())
        values = [counts[h] for h in hours]

        # Create a Plotly bar plot
        fig = px.bar(x=hours, y=values,
                     labels={"x": "Stunde des Tages", "y": "Anzahl Beobachtungen"},
                     color_discrete_sequence=["#B5EAD7"]
                     )

        fig.update_layout(
            title={
                'text': "Heutige Beobachtungen pro Stunde",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#333333"
            ),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=50, b=40),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            )
        )

        fig.update_xaxes(
            showline=True, linewidth=1, linecolor='#cccccc',
            tickfont=dict(color='#555555', size=11),
            gridcolor='#eeeeee',
            showgrid=True
        )
        fig.update_yaxes(
            showline=True, linewidth=1, linecolor='#cccccc',
            tickfont=dict(color='#555555', size=11),
            gridcolor='#eeeeee',
            showgrid=False
        )

        # Customize hover text
        fig.update_traces(
            hovertemplate="<b>Stunde %{x}</b><br>Beobachtungen: %{y}<extra></extra>")

        # Add a simple check for no data
        if not any(values):
            fig.update_layout(
                xaxis_showticklabels=False,
                yaxis_showticklabels=False,
                annotations=[
                    dict(
                        text="Noch keine Beobachtungen heute",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="#888888")
                    )
                ]
            )

        return dcc.Loading(
            type="circle",
            children=dcc.Graph(figure=fig, config={'displayModeBar': False})
        )

    # -----------------------------
    # Flask Server and Routes
    # -----------------------------
    # Create Flask server without overriding the static asset defaults.
    server = Flask(__name__)

    def setup_web_routes(app_server):
        # Route to serve images from the output directory.
        def serve_image(filename):
            image_path = os.path.join(output_dir, filename)
            if not os.path.exists(image_path):
                return "Image not found", 404
            return send_from_directory(output_dir, filename)
        app_server.route("/images/<path:filename>")(serve_image)
        app_server.route("/video_feed")(lambda: Response(
            generate_video_feed(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        ))
    setup_web_routes(server)

    # -------------------------------------
    # Dash App Setup and Google Analytics Integration
    # -------------------------------------
    external_stylesheets = [dbc.themes.BOOTSTRAP]

    # --- Cookiebot Integration ---
    cookiebot_cbid = config["COOKIEBOT_CBID"]

    cookiebot_snippet = ""  # Initialize empty
    if cookiebot_cbid:
        logger.info(f"Integrating Cookiebot with CBID: {cookiebot_cbid}")
        cookiebot_snippet = f"""
        <script id="Cookiebot" src="https://consent.cookiebot.com/uc.js" data-cbid="{cookiebot_cbid}" type="text/javascript" async></script>
        """
    else:
        logger.warning("COOKIEBOT_CBID not found in config. Cookiebot snippet will NOT be included.")
        # cookiebot_snippet = ""  # Optional comment

    # --- Google Analytics Integration ---
    ga_measurement_id = config["GA_MEASUREMENT_ID"]

    ga_snippet = ""  # Initialize empty
    if ga_measurement_id and ga_measurement_id != "G-REPLACE-ME-XXXXXX":  # Check against placeholder if using that default
        logger.info(f"Integrating Google Analytics with Measurement ID: {ga_measurement_id}")
        ga_snippet = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={ga_measurement_id}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());

          gtag('config', '{ga_measurement_id}');
          // Note: Cookiebot should handle consent for GA if configured correctly in Cookiebot backend
        </script>
        """
    else:
         logger.warning("GA_MEASUREMENT_ID not configured or is placeholder. GA snippet omitted.")

    # --- Define the custom HTML structure for Dash ---
    # PLACE COOKIEBOT *BEFORE* GOOGLE ANALYTICS
    custom_index_string = f'''
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title> 
            {{%favicon%}}
            {{%css%}}

            {cookiebot_snippet}
            {ga_snippet}
            </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    '''

    # --- Initialize Dash App with the custom index_string ---
    app = Dash(__name__, server=server, external_stylesheets=external_stylesheets,
               assets_folder=os.path.join(os.getcwd(), "assets"),
               assets_url_path="/assets",
               index_string=custom_index_string
               )
    app.config.suppress_callback_exceptions = True

    def stream_layout():
        """Layout for the live stream page using CSS classes."""
        # --- Get data needed for the new daily summaries ---
        today_date_iso = datetime.now().strftime("%Y-%m-%d")
        images_by_date = get_captured_images_by_date()
        images_for_today = images_by_date.get(today_date_iso, [])

        # --- Generate Content ---
        agreement_content = generate_daily_fused_summary_agreement(today_date_iso, images_for_today)
        weighted_content = generate_daily_fused_summary_weighted(today_date_iso, images_for_today)

        # Generate today's paginated gallery content
        todays_paginated_gallery_content = generate_subgallery(today_date_iso, page=1, include_header=False)

        # --- Build the Layout List ---
        layout_children = [
            generate_navbar(),
            dbc.Row(dbc.Col(html.H1("Live Stream", className="text-center"), width=12, className="my-3")),
            dbc.Row(dbc.Col(dcc.Loading(id="loading-video", type="default", children=html.Img(id="video-feed", src="/video_feed", className="video-feed-image")), width=12), className="my-3"),

            dbc.Row(dbc.Col(html.H2("Tagesübersicht: Agreement & Produkt-Score", className="text-center"), width=12, className="mt-4")),
            dbc.Row(dbc.Col(dcc.Loading(type="circle", children=agreement_content), width=12), className="mb-3"),

            dbc.Row(dbc.Col(html.H2(f"Tagesübersicht: Gewichteter Score, α={FUSION_ALPHA}", className="text-center"), width=12, className="mt-4")),
            dbc.Row(dbc.Col(dcc.Loading(type="circle", children=weighted_content), width=12), className="mb-5"),

            dbc.Row(dbc.Col(generate_hourly_detection_plot(), width=12), className="my-3"),

            dbc.Row(dbc.Col(html.H2("Alle heutigen Bilder", className="text-center mt-4"), width=12)),
            dbc.Row(dbc.Col(todays_paginated_gallery_content, width=12), className="my-3")
        ]
        return dbc.Container(layout_children, fluid=True)

    def gallery_layout():
        """Layout for the gallery page (calls generate_gallery which uses classes)."""
        return generate_gallery()

    # --- App Layout Modification ---
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id='auth-status-store', storage_type='session', data={'authenticated': False}),  # Stores auth flag
        dcc.Store(id='edit-target-date-store', storage_type='memory'),  # Temp store for target date
        html.Div(id="page-content"),  # Main page content
        html.Div(id="modal-container"),  # Container for the password modal
        # Other stores/hidden divs if needed
        dcc.Store(id="scroll-trigger-store", data=None),
        html.Div(id="dummy-clientside-output-div", style={"display": "none"})
    ])

    # -----------------------------
    # Dash Callbacks
    # -----------------------------
    # --- ADD Password Modal Structure ---
    @app.callback(
        Output("modal-container", "children"),
        Input("url", "pathname")  # Trigger whenever URL changes to ensure modal is added
    )
    def add_password_modal(_):  # We don't need the pathname here, just need to trigger
        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Passwort erforderlich")),
                dbc.ModalBody([
                    dbc.Alert("Bitte geben Sie das Passwort ein, um diese Seite zu bearbeiten.", color="info", id="password-modal-message"),
                    dbc.Input(id="password-input", type="password", placeholder="Passwort eingeben..."),
                ]),
                dbc.ModalFooter(
                    dbc.Button("Bestätigen", id="submit-password-button", color="primary", n_clicks=0)
                ),
            ],
            id="password-modal",
            is_open=False,  # Initially closed
            backdrop="static",  # Prevent closing by clicking outside
            keyboard=True,  # Allow closing with Esc key (might want to disable if annoying)
        )

    # --- Callback to Open Password Modal ---
    @app.callback(
        Output("password-modal", "is_open", allow_duplicate=True),
        Output("edit-target-date-store", "data"),
        Output("password-modal-message", "children", allow_duplicate=True),
        Output("password-input", "value"),  # Clear password input
        Input({'type': 'open-edit-modal-button', 'date': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def open_password_modal(n_clicks):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks) or all(c == 0 for c in n_clicks if c is not None):
            raise PreventUpdate

        # Get the date from the button that was clicked
        button_id = ctx.triggered_id
        if isinstance(button_id, dict) and button_id.get("type") == "open-edit-modal-button":
            target_date = button_id.get("date")
            if target_date:
                # Reset message and open modal
                message = dbc.Alert("Bitte geben Sie das Passwort ein, um diese Seite zu bearbeiten.", color="info")
                return True, {'date': target_date}, message, ""  # Open modal, store date, set message, clear input

        raise PreventUpdate

    # --- Callback to Check Password and Redirect (WITH DEBUG LOGGING) ---
    @app.callback(
        Output("password-modal", "is_open", allow_duplicate=True),
        Output("auth-status-store", "data"),
        Output("url", "pathname"),  # Output to trigger navigation
        Output("password-modal-message", "children", allow_duplicate=True),  # Show error message
        Input("submit-password-button", "n_clicks"),
        State("password-input", "value"),
        State("edit-target-date-store", "data"),
        State("auth-status-store", "data"),  # Get current auth status
        prevent_initial_call=True
    )
    def check_password(n_clicks, entered_password, target_date_data, auth_data):
        if n_clicks == 0 or n_clicks is None:
            logger.warning("check_password: Callback triggered but n_clicks is 0 or None.")
            raise PreventUpdate  # No actual click submission

        if not target_date_data:
            logger.error("check_password: Target date data is missing from store.")
            # Provide feedback in modal?
            return True, no_update, no_update, dbc.Alert("Error: Target date not found.", color="danger")

        target_date = target_date_data.get('date')
        if not target_date:
             logger.error("check_password: Target date missing within data.")
             return True, no_update, no_update, dbc.Alert("Error: Target date invalid.", color="danger")

        # Check if password was entered
        if not entered_password:
            logger.warning("check_password: No password entered by user.")
            return True, no_update, no_update, dbc.Alert("Bitte Passwort eingeben.", color="warning")

        # --- Perform the comparison ---
        # Use .strip() to handle potential accidental whitespace
        password_match = False
        if EDIT_PASSWORD and entered_password:
             password_match = entered_password.strip() == EDIT_PASSWORD.strip()

        if password_match:
            # Correct password
            logger.info(f"Password correct for editing date: {target_date}. Redirecting...")
            new_auth_data = {'authenticated': True}
            redirect_path = f"/edit/{target_date}"
            # Close modal, update auth store, redirect, reset message (no_update)
            return False, new_auth_data, redirect_path, no_update
        else:
            # Incorrect password
            logger.warning(f"Incorrect password entered for editing date: {target_date}")
            error_message = dbc.Alert("Falsches Passwort!", color="danger")
            # Keep modal open, don't change auth store, don't redirect, show error
            return True, no_update, no_update, error_message

    @app.callback(
        Output({"type": "daily-fused-agreement-modal", "index": ALL}, "is_open"),
        [Input({'type': 'daily-fused-agreement-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'daily-fused-agreement-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'daily-fused-agreement-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "daily-fused-agreement-modal", "index": ALL}, "is_open")]
    )
    def toggle_daily_fused_agreement_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='daily-fused-agreement-thumbnail',
            close_button_trigger_type='daily-fused-agreement-close',
            close_image_trigger_type='daily-fused-agreement-modal-image',
            thumbnail_clicks=thumbnail_clicks, close_clicks=close_clicks, modal_image_clicks=modal_image_clicks, current_states=current_states
        )

    @app.callback(
        Output({"type": "daily-fused-weighted-modal", "index": ALL}, "is_open"),
        [Input({'type': 'daily-fused-weighted-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'daily-fused-weighted-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'daily-fused-weighted-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "daily-fused-weighted-modal", "index": ALL}, "is_open")]
    )
    def toggle_daily_fused_weighted_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='daily-fused-weighted-thumbnail',
            close_button_trigger_type='daily-fused-weighted-close',
            close_image_trigger_type='daily-fused-weighted-modal-image',
            thumbnail_clicks=thumbnail_clicks, close_clicks=close_clicks, modal_image_clicks=modal_image_clicks, current_states=current_states
        )

    @app.callback(
        Output({"type": "alltime-fused-agreement-modal", "index": ALL}, "is_open"),
        [Input({'type': 'alltime-fused-agreement-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-fused-agreement-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-fused-agreement-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "alltime-fused-agreement-modal", "index": ALL}, "is_open")]
    )
    def toggle_alltime_fused_agreement_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='alltime-fused-agreement-thumbnail',
            close_button_trigger_type='alltime-fused-agreement-close',
            close_image_trigger_type='alltime-fused-agreement-modal-image',
            thumbnail_clicks=thumbnail_clicks, close_clicks=close_clicks, modal_image_clicks=modal_image_clicks, current_states=current_states
        )

    @app.callback(
        Output({"type": "alltime-fused-weighted-modal", "index": ALL}, "is_open"),
        [Input({'type': 'alltime-fused-weighted-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-fused-weighted-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-fused-weighted-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "alltime-fused-weighted-modal", "index": ALL}, "is_open")]
    )
    def toggle_alltime_fused_weighted_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='alltime-fused-weighted-thumbnail',
            close_button_trigger_type='alltime-fused-weighted-close',
            close_image_trigger_type='alltime-fused-weighted-modal-image',
            thumbnail_clicks=thumbnail_clicks, close_clicks=close_clicks, modal_image_clicks=modal_image_clicks, current_states=current_states
        )

    @app.callback(
        Output({"type": "alltime-detector-modal", "index": ALL}, "is_open"),
        [Input({'type': 'alltime-detector-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-detector-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-detector-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "alltime-detector-modal", "index": ALL}, "is_open")]
    )
    def toggle_alltime_detector_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='alltime-detector-thumbnail',
            close_button_trigger_type='alltime-detector-close',
            close_image_trigger_type='alltime-detector-modal-image',
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states
        )

    @app.callback(
        Output({"type": "alltime-classifier-modal", "index": ALL}, "is_open"),
        [Input({'type': 'alltime-classifier-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-classifier-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'alltime-classifier-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "alltime-classifier-modal", "index": ALL}, "is_open")]
    )
    def toggle_alltime_classifier_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='alltime-classifier-thumbnail',
            close_button_trigger_type='alltime-classifier-close',
            close_image_trigger_type='alltime-classifier-modal-image',
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states
        )

    # --- Callback to Display Page ---
    @app.callback(
        Output("page-content", "children"),
        Output("scroll-trigger-store", "data"),
        Input("url", "pathname"),
        Input("url", "search"),
        State("auth-status-store", "data"), # <-- ADD Auth Status State
        prevent_initial_call=True
    )
    def display_page(pathname, search, auth_data):
        scroll_trigger = no_update
        ctx = callback_context
        today_date_iso = datetime.now().strftime("%Y-%m-%d")

        is_subgallery_page_nav = (
                pathname is not None and pathname.startswith("/gallery/") and
                ctx.triggered and ctx.triggered[0]['prop_id'] == "url.search"
        )
        if is_subgallery_page_nav:
            scroll_trigger = time.time()

        if pathname is not None and pathname.startswith("/edit/"):
            date_str_iso = pathname.split("/")[-1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str_iso):
                 return "Invalid date format.", no_update

            # CHECK 1: Authentication
            if not auth_data or not auth_data.get('authenticated'):
                logger.warning(f"Unauthorized attempt to access edit page: {pathname}")
                return dbc.Container([  # Simple access denied page
                    generate_navbar(),
                    html.H2("Zugriff verweigert", className="text-danger text-center mt-4"),
                    html.P("Sie müssen authentifiziert sein, um diese Seite anzuzeigen.", className="text-center"),
                    dbc.Button("Zurück zur Galerie", href="/gallery", color="primary")
                ], fluid=True), no_update

            # CHECK 2: Prevent editing today's data
            if date_str_iso == today_date_iso:
                logger.warning(f"Authenticated user attempted to access edit page for current day: {date_str_iso}")
                return dbc.Container([
                     generate_navbar(), html.H2(f"Edit Bilder vom {date_str_iso}", className="text-center my-3"),
                     dbc.Alert("Die Bearbeitung der Galerie für den aktuellen Tag ist nicht erlaubt.", color="warning"),
                     dbc.Button("Back to Subgallery", href=f"/gallery/{date_str_iso}", color="secondary", className="me-2"),
                     dbc.Button("Back to Main Gallery", href="/gallery", color="secondary"),
                 ], fluid=True), no_update

            # If authenticated AND not today, generate the edit page
            logger.info(f"Authorized access to edit page: {pathname}")
            return generate_edit_page(date_str_iso), no_update

        # --- SUBGALLERY ROUTE ---
        elif pathname is not None and pathname.startswith("/gallery/"):
            date = pathname.split("/")[-1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                 return "Invalid date format.", no_update
            page = 1
            if search:
                params = parse_qs(search.lstrip('?'))
                if 'page' in params:
                    try: page = int(params['page'][0])
                    except ValueError: page = 1
            content = generate_subgallery(date, page)
            return content, scroll_trigger
        # --- OTHER ROUTES ---
        elif pathname == "/gallery": return generate_gallery(), no_update
        elif pathname == "/species": return species_summary_layout(), no_update
        elif pathname == "/" or pathname is None or pathname == "": return stream_layout(), no_update
        # --- 404 ---
        else:
            logger.warning(f"404 Not Found for pathname: {pathname}")
            return dbc.Container([
                generate_navbar(), html.H1("404 - Page Not Found", className="text-center text-danger mt-5"),
                html.P(f"The path '{pathname}' was not recognized.", className="text-center"),
                dbc.Button("Go to Homepage", href="/", color="primary")
            ], fluid=True), no_update

    # Callback to update the selected images store based on checkboxes
    @app.callback(
        Output('selected-images-store', 'data'),
        Input({'type': 'edit-image-checkbox', 'index': ALL}, 'value'),  # Triggered by checkbox value changes
        State({'type': 'edit-image-checkbox', 'index': ALL}, 'id'),
        prevent_initial_call=True
    )
    def update_selected_images(checkbox_values, checkbox_ids):
        # This callback runs whenever a checkbox's value changes (natively or via JS .click())
        # print("update_selected_images triggered by checkbox value change")
        selected_paths = []
        if not checkbox_ids:
            return []
        # checkbox_values and checkbox_ids should correspond index-wise
        for i, cb_id in enumerate(checkbox_ids):
            is_checked = checkbox_values[i] if i < len(checkbox_values) else False
            if is_checked:
                relative_path = cb_id['index']  # Get the path from the ID
                selected_paths.append(relative_path)
        # print(f"Updating selected-images-store with: {selected_paths}")
        return selected_paths

    # Callback to trigger delete confirmation
    @app.callback(
        Output('confirm-delete', 'displayed'),
        Input('delete-button', 'n_clicks'),
        Input('delete-button-bottom', 'n_clicks'),  # Trigger from bottom button too
        prevent_initial_call=True,
    )
    def display_delete_confirm(n_clicks_top, n_clicks_bottom):
        if (n_clicks_top and n_clicks_top > 0) or (n_clicks_bottom and n_clicks_bottom > 0) :
            return True
        return False

    # Callback to handle deletion after confirmation
    @app.callback(
        Output('edit-status-message', 'children'),
        Input('confirm-delete', 'submit_n_clicks'),
        State('selected-images-store', 'data'),
        State('url', 'pathname'),  # Get the date from the current URL
        prevent_initial_call=True,
    )
    def handle_delete(submit_n_clicks, selected_images, pathname):
        if not submit_n_clicks or submit_n_clicks == 0:
            raise PreventUpdate # No submission yet

        if not selected_images:
            return dbc.Alert("No images selected for deletion.", color="warning"), no_update  # Or just no_update

        # Extract date from pathname like /edit/YYYY-MM-DD
        match = re.search(r"/edit/(\d{4}-\d{2}-\d{2})", pathname)
        if not match:
            return dbc.Alert("Error: Could not determine date from URL.", color="danger"), no_update
        date_str_iso = match.group(1)

        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return dbc.Alert(f"Error: Could not read CSV data for {date_str_iso} to perform deletion.", color="danger"), no_update

        # Identify rows to keep
        # We stored relative paths (folder/file) in selected_images
        # Need to match based on the filename part
        selected_filenames = {os.path.basename(p) for p in selected_images}
        df_to_keep = df[~df['optimized_name'].isin(selected_filenames)]

        # Write the filtered DataFrame back
        success_csv = write_csv_for_date(date_str_iso, df_to_keep)

        deleted_files_count = 0
        error_messages = []

        # Delete image files
        for relative_path in selected_images:
            if not delete_image_files(relative_path):
                 error_messages.append(f"Failed to delete one or more files for {os.path.basename(relative_path)}")

        # --- Feedback Message ---
        status_messages = []
        if success_csv:
            status_messages.append(f"Successfully updated CSV for {date_str_iso}.")
            status_messages.append(f"{len(selected_images)} entries removed.")
        else:
             status_messages.append(f"Error updating CSV for {date_str_iso}.")

        if deleted_files_count > 0 or not error_messages :
             status_messages.append(f"Attempted deletion of files for {len(selected_images)} entries.") # Be slightly vague if some failed
        if error_messages:
            status_messages.extend(error_messages)

        alert_color = "success" if success_csv and not error_messages else ("warning" if success_csv else "danger")

        return dbc.Alert(html.Ul([html.Li(msg) for msg in status_messages]), color=alert_color, dismissable=True)


    # Callback to handle download request
    @app.callback(
        Output('download-zip', 'data'),
        Output('edit-status-message', 'children', allow_duplicate=True),  # Update status too
        Input('download-button', 'n_clicks'),
        Input('download-button-bottom', 'n_clicks'),
        State('selected-images-store', 'data'),
        State('url', 'pathname'),
        prevent_initial_call=True,
    )
    def handle_download(n_clicks_top, n_clicks_bottom, selected_images, pathname):
        triggered = callback_context.triggered_id == 'download-button' or callback_context.triggered_id == 'download-button-bottom'
        if not triggered:
             raise PreventUpdate

        if not selected_images:
            return no_update, dbc.Alert("No images selected for download.", color="warning", dismissable=True)

        # Extract date from pathname
        match = re.search(r"/edit/(\d{4}-\d{2}-\d{2})", pathname)
        if not match:
            return no_update, dbc.Alert("Error: Could not determine date from URL.", color="danger", dismissable=True)
        date_str_iso = match.group(1)

        df = read_csv_for_date(date_str_iso)
        if df.empty:
             return no_update, dbc.Alert(f"Error: Could not read CSV data for {date_str_iso} to perform download.", color="danger", dismissable=True)

        # --- 1. Update CSV ---
        download_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_filenames = {os.path.basename(p) for p in selected_images}

        # Ensure the column exists
        if 'downloaded_timestamp' not in df.columns:
            df['downloaded_timestamp'] = ''
            df['downloaded_timestamp'] = df['downloaded_timestamp'].astype(str)


        # Update rows - use .loc for safer assignment
        rows_to_update = df['optimized_name'].isin(selected_filenames)
        df.loc[rows_to_update, 'downloaded_timestamp'] = download_timestamp

        success_csv = write_csv_for_date(date_str_iso, df)

        # --- 2. Create Zip ---
        zip_buffer = io.BytesIO()
        errors_zipping = []
        files_added = 0
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for relative_path in selected_images:
                # We want the ORIGINAL file for download
                original_relative_path = relative_path.replace("_optimized", "_original")
                absolute_original_path = os.path.join(output_dir, original_relative_path)
                if os.path.exists(absolute_original_path):
                    try:
                        # Add file to zip, using the original filename as the archive name
                        zip_file.write(absolute_original_path, arcname=os.path.basename(original_relative_path))
                        files_added += 1
                    except Exception as e:
                        logger.error(f"Error adding {absolute_original_path} to zip: {e}")
                        errors_zipping.append(os.path.basename(original_relative_path))
                else:
                     logger.warning(f"Original file not found for download: {absolute_original_path}")
                     errors_zipping.append(os.path.basename(original_relative_path) + " (Not Found)")

        zip_buffer.seek(0)

        # --- 3. Prepare Download Data ---
        zip_data = base64.b64encode(zip_buffer.read()).decode('utf-8')
        download_filename = f"watchmybirds_{date_str_iso.replace('-','')}_download.zip"
        download_dict = dict(content=zip_data, filename=download_filename, base64=True, type='application/zip')

        # --- 4. Prepare Status Message ---
        status_messages = []
        if success_csv:
            status_messages.append(f"Marked {len(selected_images)} images as downloaded in CSV for {date_str_iso}.")
        else:
            status_messages.append(f"Error updating CSV for {date_str_iso}.")

        if files_added > 0:
             status_messages.append(f"Prepared {files_added} original images for download.")
        if errors_zipping:
            status_messages.append(f"Could not add {len(errors_zipping)} files to the zip: {', '.join(errors_zipping[:3])}{'...' if len(errors_zipping)>3 else ''}")

        alert_color = "success" if success_csv and files_added > 0 and not errors_zipping else "warning"

        # Send download data and status message
        return download_dict, dbc.Alert(html.Ul([html.Li(msg) for msg in status_messages]), color=alert_color, dismissable=True)

    def _toggle_modal_generic(
            # Pass the specific types expected for this modal group
            open_trigger_type: str,
            close_button_trigger_type: str,
            close_image_trigger_type: str,
            thumbnail_clicks,
            close_clicks,
            modal_image_clicks,
            current_states
    ):
        """Generic function to toggle modals based on explicit trigger types."""
        ctx = callback_context
        # Initialize variables used in the try block
        triggered_prop = None
        triggered_value = None
        triggered_id_dict = None
        triggered_type = None
        triggered_component_index = None

        # Guard clause: No trigger or no outputs defined
        if not ctx.triggered or not ctx.outputs_list:
            # Returning no_update for all outputs is often preferred over raising PreventUpdate
            # when dealing with ALL pattern callbacks if some outputs might exist.
            num_outputs = len(ctx.outputs_list) if ctx.outputs_list else (len(current_states) if current_states else 0)
            if num_outputs > 0:
                return [no_update] * num_outputs
            else:
                # If truly nothing to update, PreventUpdate might be applicable,
                # but returning an empty list or handling upstream might be safer.
                # For now, let's stick to no_update based on state length if possible.
                if current_states is not None: return [no_update] * len(current_states)
                return []  # Or potentially raise PreventUpdate if appropriate for your broader app structure

        triggered_prop = ctx.triggered[0]['prop_id']
        triggered_value = ctx.triggered[0]['value']

        # Guard clause: Trigger value indicates no actual click/event
        if triggered_value is None or triggered_value == 0:
            return [no_update] * len(ctx.outputs_list)

        # Safely parse the trigger ID
        try:
            triggered_id_str = triggered_prop.split('.')[0]
            triggered_id_dict = json.loads(triggered_id_str)
            triggered_type = triggered_id_dict.get("type")
            triggered_component_index = triggered_id_dict.get("index")
        except (IndexError, json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.error(f"Error parsing trigger ID: {triggered_prop} -> {e}", exc_info=True)
            return [no_update] * len(ctx.outputs_list)

        # Guard clause: Parsed ID lacks necessary parts
        if triggered_type is None or triggered_component_index is None:
            logger.error(f"Invalid trigger type/index after parsing: {triggered_id_dict}")
            return [no_update] * len(ctx.outputs_list)

        output_component_ids = [output['id'] for output in ctx.outputs_list]
        new_states = [no_update] * len(output_component_ids)

        # Find the index in the output list corresponding to the triggered component's index
        target_list_index = -1
        for i, output_id in enumerate(output_component_ids):
            if isinstance(output_id, dict) and output_id.get("index") == triggered_component_index:
                target_list_index = i
                break

        # Guard clause: Could not find the target modal in the output list
        if target_list_index == -1:
            logger.error(f"Could not find target modal index for trigger component index {triggered_component_index}")
            return new_states  # Return no_update for all

        # Determine the expected type of the output modal component
        try:
            expected_modal_output_type = ctx.outputs_list[0]['id']['type']
        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Could not determine expected output type from context: {e}", exc_info=True)
            return [no_update] * len(ctx.outputs_list)

        # Guard clause: Ensure state list matches output list
        if current_states is None or len(current_states) != len(output_component_ids):
            logger.error(
                f"Mismatch or missing current_states. Len states: {len(current_states) if current_states else 'None'}, Len outputs: {len(output_component_ids)}"
            )
            return [no_update] * len(output_component_ids)

        # --- Main Logic ---
        target_output_id = output_component_ids[target_list_index]

        # Check if the target output component matches the expected type for this modal group
        if not isinstance(target_output_id, dict) or target_output_id.get("type") != expected_modal_output_type:
            logger.error(
                f"Output type mismatch at target index {target_list_index}. Expected '{expected_modal_output_type}', Found '{target_output_id.get('type')}'")
            return [no_update] * len(output_component_ids)  # Prevent update on mismatch

        is_currently_open = current_states[target_list_index]

        # --- Logic for Opening ---
        if triggered_type == open_trigger_type:
            if not is_currently_open:
                # Explicitly create the list of desired states: close all, open target
                final_states = [False] * len(output_component_ids)
                final_states[target_list_index] = True
                new_states = final_states
            # else: modal already open, do nothing (implicit no_update)

        # --- Logic for Closing ---
        elif triggered_type == close_button_trigger_type or triggered_type == close_image_trigger_type:
            if is_currently_open:  # Only close if it's open
                # Explicitly create the list: keep existing states, close target
                final_states = list(current_states)  # Important: work from current state
                final_states[target_list_index] = False
                new_states = final_states
            # else: modal already closed, do nothing (implicit no_update)

        # else: trigger type didn't match open/close types, do nothing (implicit no_update)

        return new_states

    @app.callback(
        Output({"type": "subgallery-modal-modal", "index": ALL}, "is_open"),
        [Input({'type': 'subgallery-thumbnail', 'index': ALL}, 'n_clicks'),
         Input({'type': 'subgallery-modal-close', 'index': ALL}, 'n_clicks'),
         Input({'type': 'subgallery-modal-modal-image', 'index': ALL}, 'n_clicks')],
        [State({"type": "subgallery-modal-modal", "index": ALL}, "is_open")]
    )
    def toggle_subgallery_modal(thumbnail_clicks, close_clicks, modal_image_clicks, current_states):
        return _toggle_modal_generic(
            open_trigger_type='subgallery-thumbnail',
            close_button_trigger_type='subgallery-modal-close',
            close_image_trigger_type='subgallery-modal-modal-image',
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states
        )

    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='scrollToPagination'
        ),
        # Change this Output:
        Output("dummy-clientside-output-div", "children"),  # Target the dummy div instead
        Input("scroll-trigger-store", "data"),
        prevent_initial_call=True
    )

    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    def run(debug=False, host="0.0.0.0", port=8050):
        logger.info(f"Starting Dash server on http://{host}:{port}")
        app.run_server(host=host, port=port, debug=debug)

    return {"app": app, "server": server, "run": run}
