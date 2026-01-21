from __future__ import annotations

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from logging_config import get_logger

logger = get_logger(__name__)


class FrameGenerator:
    """Generates JPEG-encoded frames with timestamps."""

    def __init__(self, video_capture):
        """Stores a reference to the VideoCapture instance."""
        self.video_capture = video_capture

    def _draw_timestamp(self, image, padding_x_percent, padding_y_percent):
        """Adds a timestamp to the image."""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _generate_placeholder(self, placeholder, padding_x_percent, padding_y_percent):
        """Generates a placeholder with noise and a timestamp."""
        placeholder_copy = placeholder.copy()
        noise = np.random.randint(-100, 20, placeholder_copy.shape, dtype=np.int16)
        placeholder_copy = np.clip(
            placeholder_copy.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)
        return self._draw_timestamp(
            placeholder_copy, padding_x_percent, padding_y_percent
        )

    def generate_frames(self, output_resize_width, stream_fps):
        """Continuously yields JPEG-encoded frames."""
        static_placeholder_path = "assets/static_placeholder.jpg"
        if os.path.exists(static_placeholder_path):
            static_placeholder = cv2.imread(static_placeholder_path)
            if static_placeholder is not None:
                original_h, original_w = static_placeholder.shape[:2]
                ratio = original_h / float(original_w)
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * ratio)
                static_placeholder = cv2.resize(
                    static_placeholder, (placeholder_w, placeholder_h)
                )
            else:
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * 9 / 16)
                static_placeholder = np.zeros(
                    (placeholder_h, placeholder_w, 3), dtype=np.uint8
                )
        else:
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros(
                (placeholder_h, placeholder_w, 3), dtype=np.uint8
            )

        padding_x_percent = 0.005
        padding_y_percent = 0.04

        while True:
            start_time = time.time()
            frame = self.video_capture.get_frame()
            if frame is not None:
                w, h = self.video_capture.resolution
                output_resize_height = int(h * placeholder_w / w)
                resized_frame = cv2.resize(frame, (placeholder_w, output_resize_height))
                frame_with_ts = self._draw_timestamp(
                    resized_frame, padding_x_percent, padding_y_percent
                )
            else:
                frame_with_ts = self._generate_placeholder(
                    static_placeholder, padding_x_percent, padding_y_percent
                )

            ret, buffer = cv2.imencode(
                ".jpg", frame_with_ts, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            else:
                logger.error("Failed to encode frame.")

            elapsed = time.time() - start_time
            desired_frame_time = 1.0 / stream_fps
            if elapsed < desired_frame_time:
                time.sleep(desired_frame_time - elapsed)
