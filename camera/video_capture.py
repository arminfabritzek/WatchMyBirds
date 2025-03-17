# ------------------------------------------------------------------------------
# Video Capture Class for Input Streams
# camera/video_capture.py
# ------------------------------------------------------------------------------
# video_capture.py
from config import load_config
config = load_config()
import os
import subprocess
import cv2
import queue
import threading
from threading import Event
import time
import numpy as np
import logging
from logging_config import get_logger
logger = get_logger(__name__)

class VideoCapture:
    """
    Handles video input streams from various sources (RTSP, WebRTC, Webcam, FFmpeg).
    """
    def __init__(self, source, debug=False):
        """
        Initializes the VideoCapture object and detects the stream type.
        """
        self.source = source
        self.debug = debug  # Initialize the debug attribute
        self.stop_flag = False
        self.stop_event = Event()  # Event to signal the thread to stop
        self.cap = None
        self.ffmpeg_process = None  # Used for FFmpeg streams
        self.q = queue.Queue(maxsize=10)  # Queue for storing frames
        self.reader_thread = None  # Initialize the reader_thread attribute
        self.last_frame_time = time.time()
        self.health_check_thread = None  # Initialize the health_check_thread attribute
        self.retry_count = 0  # Initialize retry count for reinitialization
        self.stream_type = self._detect_stream_type()

        self.reinit_lock = threading.Lock()
        self.reinitializing = False

        logger.debug(f"Initialized VideoCapture with source: {self.source}, stream_type: {self.stream_type}")

        if self.stream_type == "rtsp":
            self._get_stream_resolution_ffprobe()  # Detect resolution before setting up

        self._setup_capture()
        self._start_reader_thread()
        self._start_health_check_thread()

    def _detect_stream_type(self):
        """
        Automatically detects the stream type based on the source string.

        :return: Detected stream type as a string ("rtsp", "http", "webcam").
        """
        logger.debug("Detect stream type...")
        if isinstance(self.source, str):
            if self.source.startswith("rtsp://"):
                logger.debug("Stream type detected as RTSP.")
                return "rtsp"
            elif self.source.startswith(("http://", "https://")):
                logger.debug("Stream type detected as HTTP.")
                return "http"
        elif isinstance(self.source, int):  # Integer sources are treated as webcams
            logger.debug("Stream type detected as Webcam.")
            return "webcam"
        error_msg = f"Unable to determine stream type for source: {self.source}"
        logger.error(error_msg)
        raise ValueError(f"Unable to determine stream type for source: {self.source}")


    def _setup_capture(self):
        """
        Sets up the video capture based on the detected stream type.
        """
        logger.debug("Setting up video capture...")
        if self.stream_type == "rtsp":
            self._setup_ffmpeg()
        elif self.stream_type == "http":
            self._setup_http()
        elif self.stream_type == "webcam":
            self._setup_webcam()
        else:
            error_msg = f"Unsupported stream type: {self.stream_type}"
            logger.error(error_msg)
            raise ValueError(f"Unsupported stream type: {self.stream_type}")
        logger.debug("Video capture setup completed.")

    def _setup_ffmpeg(self):
        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "50000000",
            "-analyzeduration", "10000000",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "pipe:"
        ]

        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        for attempt in range(5):  # Retry up to 5 times
            try:
                logger.debug(f"Starting FFmpeg process (attempt {attempt + 1})...")
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
                )
                logger.debug("FFmpeg process started successfully.")

                # Start logging FFmpeg errors if in debug mode
                if self.debug:
                    threading.Thread(target=self._log_ffmpeg_errors, daemon=True).start()

                # Verify FFmpeg process is running
                if self.ffmpeg_process.poll() is not None:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    error_msg = f"FFmpeg process terminated prematurely. STDERR: {stderr_output}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                return  # Exit loop on success
            except Exception as e:
                logger.error(f"Failed to start FFmpeg process: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
        error_msg = "Failed to start FFmpeg after multiple attempts."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _log_ffmpeg_errors(self):
        """
        Continuously logs FFmpeg's stderr for debugging purposes, only if debug mode is enabled.
        """
        logger.debug("Starting FFmpeg stderr logging.")
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                if line:
                    logger.debug(f"FFmpeg STDERR: {line.decode('utf-8').strip()}")
                if self.stop_event.is_set():
                    logger.debug("Stop event set. Ending FFmpeg stderr logging.")
                    break
        except Exception as e:
            logger.error(f"Exception while logging FFmpeg stderr: {e}")
        logger.debug("FFmpeg stderr logging thread terminated.")

    def _get_stream_resolution_ffprobe(self):
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.source
        ]

        logger.debug(f"Running FFprobe command: {' '.join(ffprobe_cmd)}")

        try:
            output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
            logger.debug(f"FFprobe output: {output}")
            # Parse the FFprobe output
            width, height = map(int, output.split('\n'))
            logger.debug(f"Detected stream resolution: {width}x{height}")

            # Set the resolution as instance attributes
            self.stream_width = width
            self.stream_height = height

        except subprocess.TimeoutExpired:
            logger.debug("FFprobe command timed out. Cannot reach the RTSP stream.")
            raise RuntimeError("FFprobe timed out while trying to get stream resolution.")
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode().strip()
            logger.debug(f"FFprobe failed with error: {error_output}")
            raise RuntimeError("Failed to get stream resolution using FFprobe.")
        except ValueError as ve:
            logger.debug(f"Error parsing FFprobe output: {ve}")
            raise RuntimeError("Failed to parse stream resolution from FFprobe output.")
        except Exception as e:
            logger.debug(f"Unexpected error during FFprobe: {e}")
            raise RuntimeError("An unexpected error occurred while getting stream resolution.")

    def _setup_http(self):
        """
        Sets up OpenCV's VideoCapture for HTTP streams.
        """
        logger.debug(f"Setting up HTTP stream with source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open HTTP stream: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Detected HTTP stream resolution: {self.stream_width}x{self.stream_height}")

    def _setup_webcam(self):
        """
        Sets up OpenCV's VideoCapture for webcam streams.
        """
        logger.info(f"Setting up Webcam with source index: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open webcam with index: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Detected Webcam resolution: {self.stream_width}x{self.stream_height}")

    def _start_reader_thread(self):
        """
        Starts the reader thread if it's not already running.
        Ensures only one thread is active at a time to avoid conflicts.
        """
        if self.reader_thread and self.reader_thread.is_alive():
            logger.info("Reader thread already running; skipping reinitialization")

        else:
            # Add a short delay to allow FFmpeg to initialize properly
            logger.info("Starting reader thread...")
            time.sleep(2)
            self.reader_thread = threading.Thread(target=self._reader, daemon=True, name="ReaderThread")
            self.reader_thread.start()
            logger.info("Reader thread started successfully.")

    def _start_health_check_thread(self):
        """
        Starts a health check thread to monitor FFmpeg subprocess.
        """
        if self.stream_type != "rtsp":
            logger.info("Health check thread is only applicable for RTSP streams. Skipping.")
            return

        if self.health_check_thread and self.health_check_thread.is_alive():
            logger.info("Health check thread already running; skipping reinitialization.")
        else:
            logger.info("Starting health check thread...")
            self.health_check_thread = threading.Thread(target=self._health_check, daemon=True, name="HealthCheckThread")
            self.health_check_thread.start()
            logger.info("Health check thread started successfully.")

    def _health_check(self):
        """
        Periodically checks if the stream (FFmpeg for RTSP, or OpenCV for HTTP) is alive.
        If not, attempts to reinitialize.
        """
        logger.info("Health check thread is running.")
        while not self.stop_event.is_set():
            if self.stream_type == "rtsp":
                if self.ffmpeg_process:
                    retcode = self.ffmpeg_process.poll()
                    if retcode is not None:
                        stderr_output = self.ffmpeg_process.stderr.read().decode().strip()
                        logger.error(f"FFmpeg subprocess terminated with return code {retcode}. STDERR: {stderr_output}")
                        self._reinitialize_camera(reason="RTSP FFmpeg subprocess terminated unexpectedly.")
                    else:
                        # Check if frames have been read recently.
                        if time.time() - self.last_frame_time > 10:  # 10-second threshold to wait until reinitialization.
                            logger.error("No frame received for over 10 seconds; triggering reinitialization.")
                            self._reinitialize_camera(reason="RTSP stream stale.")
            elif self.stream_type == "http":
                if not self.cap or not self.cap.isOpened():
                    logger.error("HTTP stream is not opened. Triggering reinitialization.")
                    self._reinitialize_camera(reason="HTTP stream not opened.")
            # You can also add a similar check for webcams if needed.
            time.sleep(5)  # Check every 5 seconds
        logger.info("Health check thread is stopping.")

    def _reader(self):
        """
        Continuously reads frames from the video source and places only the latest frame in the queue.
        Terminates cleanly if stop_event is set.
        """
        logger.info("Reader thread is running.")
        while not self.stop_event.is_set():  # Check for thread termination
            try:
                frame = self._read_frame()
                if self.stop_event.is_set():  # Recheck stop_event after potentially long operations
                    logger.info("Stop event set. Reader thread is terminating.")
                    break
                if frame is not None:
                    # Clear the queue before adding the new frame
                    with self.q.mutex:
                        self.q.queue.clear()
                    self.q.put(frame, timeout=0.1)  # Always have the latest frame in the queue
                else:
                    logger.info("Frame not available, triggering reinitialization.")
                    self._reinitialize_camera(reason="Received None frame.")
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                if self.stop_event.is_set():
                    break
                self._reinitialize_camera()  # Reinitialize in case of error
        logger.info("Reader thread has exited.")

    def _read_frame(self):
        if self.stream_type == "rtsp":
            return self._read_ffmpeg_frame()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.error("HTTP capture failed to read frame.")
                self._reinitialize_camera(reason="HTTP capture failed to read frame.")
                return None
        else:
            logger.error("HTTP capture is not opened.")
            self._reinitialize_camera(reason="HTTP capture not opened.")
            return None

    def _read_ffmpeg_frame(self):
        frame_size = self.stream_width * self.stream_height * 3  # For bgr24 (3 bytes per pixel)

        try:
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                logger.error(f"FFmpeg produced incomplete frame: expected {frame_size} bytes, got {len(raw_frame)} bytes.")
                return None  # Skip incomplete frames
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.stream_height, self.stream_width, 3))
            self.last_frame_time = time.time()  # Update timestamp on successful frame read
            return frame
        except Exception as e:
            logger.error(f"Error reading frame from FFmpeg: {e}")
            return None

    def _schedule_reinit(self, reason):
        delay = 2 ** self.retry_count
        logger.info(f"Scheduling reinitialization in {delay} seconds due to: {reason}")
        threading.Timer(delay, self._reinitialize_camera, kwargs={'reason': reason}).start()

    def _reinitialize_camera(self, reason="Unknown"):
        """
        Attempts to reinitialize the camera after a failure.

        :param reason: The reason triggering reinitialization.
        """
        # Try to get the lock to avoid parallel reinitializations.
        if not self.reinit_lock.acquire(blocking=False):
            logger.info("Reinitialization is already being performed in another thread.")
            return

        # Check whether a reinitialization is already activated.
        if self.reinitializing:
            logger.info("Reinitialization is already in progress. Skip retry.")
            self.reinit_lock.release()
            return

        self.reinitializing = True
        try:
            logger.info(f"Reinitializing camera due to: {reason}")
            if self.retry_count >= 5:
                logger.info("Maximum retry attempts reached. Scheduling longer delay before next attempt.")
                # Longer delay before retrying again (e.g. 60 seconds)
                threading.Timer(60, self._reinitialize_camera, kwargs={'reason': "Retry after long delay"}).start()
                # Reset the retry counter after scheduling a longer delay
                self.retry_count = 0
                return

            self.retry_count += 1
            logger.info(f"Reinitialization attempt {self.retry_count}/5.")

            # Stop current capture and threads
            self.release()
            logger.info("Waiting 2 seconds before reinitialization attempt.")
            time.sleep(2)

            self.stop_event.clear()  # Reset stop_event for new threads
            self._setup_capture()
            self._start_reader_thread()
            self._start_health_check_thread()
            logger.info("Reinitialization successful.")
            self.retry_count = 0  # Reset retry count on success
        except Exception as e:
            logger.info(f"Reinitialization failed: {e}")
            logger.info("Scheduling next reinitialization attempt.")
            self._schedule_reinit(reason=f"Failed to reinitialize: {e}")
        finally:
            self.reinitializing = False
            self.reinit_lock.release()

    def get_frame(self):
        try:
            # Block for up to 0.1 seconds waiting for a frame.
            frame = self.q.get(timeout=0.3)
            # logger.debug("Retrieved a frame from the queue.")
            return frame
        except queue.Empty:
            logger.info("Queue is empty, unable to retrieve frame.")
            return None

    def generate_frames(self, output_resize_width, stream_fps):
        """
        Generator that yields JPEG-encoded frames from the video stream.
        Dynamically scales the font size based on the image height.
        """
        import time
        import numpy as np
        import cv2
        from PIL import Image, ImageDraw, ImageFont

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

        # Percent-based padding from the bottom-right corner
        padding_x_percent = 0.005  # e.g., 1% of image width
        padding_y_percent = 0.04  # e.g., 3.5% of image height
        # Let's make the font ~3% of the image height (but at least 12)
        min_font_size = 12
        min_font_size_percent = 0.05

        while True:
            start_time = time.time()
            frame = self.get_frame()
            if frame is not None:
                # Resize the frame
                w, h = self.resolution
                output_resize_height = int(h * placeholder_w / w)
                resized_frame = cv2.resize(frame, (placeholder_w, output_resize_height))

                # Convert to PIL to draw text
                pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                # Get image dimensions
                img_width, img_height = pil_image.size

                # Convert percentages to pixel values
                padding_x = int(img_width * padding_x_percent)
                padding_y = int(img_height * padding_y_percent)

                # ----- Dynamically scale the font size -----
                scaled_font_size = max(min_font_size, int(img_height * min_font_size_percent))

                try:
                    custom_font = ImageFont.truetype("assets/WRP_cruft.ttf", scaled_font_size)
                except IOError:
                    custom_font = ImageFont.load_default()
                # -------------------------------------------

                # Prepare timestamp text
                timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")

                # Calculate the text width and height (for offset calculation)
                bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Bottom-right position, offset by padding
                text_x = img_width - text_width - padding_x
                text_y = img_height - text_height - padding_y

                # Draw the timestamp (no background box)
                draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")

                # Convert back to OpenCV image
                resized_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.error("Failed to encode resized frame.")
            else:
                # Use placeholder if no frame is available
                placeholder_copy = static_placeholder.copy()
                noise = np.random.randint(-100, 20, placeholder_copy.shape, dtype=np.int16)
                placeholder_copy = np.clip(placeholder_copy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                pil_image = Image.fromarray(cv2.cvtColor(placeholder_copy, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                img_width, img_height = pil_image.size
                padding_x = int(img_width * padding_x_percent)
                padding_y = int(img_height * padding_y_percent)

                scaled_font_size = max(min_font_size, int(img_height * min_font_size_percent))
                try:
                    custom_font = ImageFont.truetype("assets/WRP_cruft.ttf", scaled_font_size)
                except IOError:
                    custom_font = ImageFont.load_default()

                timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
                bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = img_width - text_width - padding_x
                text_y = img_height - text_height - padding_y
                draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")

                placeholder_copy = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', placeholder_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.error("Failed to encode placeholder frame.")

            # Maintain the requested FPS
            elapsed = time.time() - start_time
            desired_frame_time = 1.0 / stream_fps
            if elapsed < desired_frame_time:
                time.sleep(desired_frame_time - elapsed)

    def release(self):
        """
        Releases the video capture resources.
        """
        logger.info("Releasing resources...")
        self.stop_event.set()  # Signal the thread to stop
        self.stop_flag = True

        # Stop the reader thread if it's not the current thread
        if self.reader_thread and self.reader_thread.is_alive():
            if self.reader_thread != threading.current_thread():
                logger.info("Joining reader thread...")
                self.reader_thread.join(timeout=5)  # Wait for the thread to exit
                if self.reader_thread.is_alive():
                    logger.info("Reader thread did not terminate within timeout.")
            else:
                logger.info("Skipping join on current thread (reader thread).")

        # Clear the queue
        with self.q.mutex:
            self.q.queue.clear()
            logger.info("Frame queue cleared.")

        # Stop the health check thread if it's not the current thread
        if self.health_check_thread and self.health_check_thread != threading.current_thread():
            logger.info("Joining health check thread...")
            self.health_check_thread.join(timeout=5)
            if self.health_check_thread.is_alive():
                logger.info("Health check thread did not terminate within timeout.")
        else:
            logger.info("Skipping join on current thread (health check thread).")

        # Release FFmpeg process
        if self.ffmpeg_process:
            logger.info("Terminating FFmpeg process...")
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)  # Wait for FFmpeg to exit
                logger.info("FFmpeg process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.info("FFmpeg process did not terminate in time. Killing process...")
                self.ffmpeg_process.kill()
                logger.info("FFmpeg process killed.")

        # Release OpenCV capture if used
        if self.cap:
            logger.info("Releasing OpenCV VideoCapture.")
            self.cap.release()

        logger.info("Resources released.")

    @property
    def resolution(self):
        """
        Property to get the resolution of the video stream.
        """
        return (self.stream_width, self.stream_height)