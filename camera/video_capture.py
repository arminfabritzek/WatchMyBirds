import cv2
import queue
import threading
import time


class VideoCapture:
    """
    A class for threaded video capture, suitable for specialized applications requiring minimal latency.
    """

    _counter = 0  # Class-level counter for multiple camera instances

    def __init__(self, source=0, backend=None):
        """
        Initializes the video capture object with threading and queue.

        :param source: Camera source, such as an index for a connected camera (e.g., 0 for the default camera).
        :param backend: Optional backend for OpenCV (e.g., cv2.CAP_FFMPEG).
        """
        # Increment instance counter for multiple cameras
        VideoCapture._counter += 1
        self.num_instance = VideoCapture._counter

        self.backend = backend
        self.source = source
        self.stop_flag = False  # Flag to stop the thread

        # Initialize VideoCapture with optional backend
        if self.backend is not None:
            self.cap = cv2.VideoCapture(self.source, self.backend)
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout

        if not self.cap.isOpened():
            raise ValueError("Failed to open video source.")

        # Queue to store frames
        self.q = queue.Queue(maxsize=10)  # Limit to 10 frames
        self.counter = 0  # Counter for dropped frames

        # Start the reader thread
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        """
        Background thread to read frames continuously and keep only the latest frame.
        """
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from video source.")
                self.counter += 1
                time.sleep(1)  # Prevent busy looping
                if self.counter >= 10:  # Attempt to reinitialize after 10 failed reads
                    print("Reinitializing camera after multiple failures...")
                    self._reinitialize_camera()
                continue

            # Discard old frames if queue is full
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            # Add the new frame to the queue
            try:
                self.q.put(frame, timeout=0.1)
            except queue.Full:
                print("Frame queue is full. Dropping frame.")

    def _reader(self):
        """
        Background thread to read frames continuously and keep only the latest frame.
        """
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from video source.")
                self.counter += 1
                if self.counter >= 10:  # Attempt to reinitialize after 10 failed reads
                    print("Reinitializing camera after multiple failures...")
                    self._reinitialize_camera()
                time.sleep(1)  # Prevent busy looping
                continue

            self.counter = 0  # Reset failure counter on successful frame read

            # Discard old frames if queue is full
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    print("Queue unexpectedly empty during discard operation.")

            # Add the new frame to the queue
            try:
                self.q.put(frame, timeout=0.1)
            except queue.Full:
                print("Frame queue is full. Dropping frame.")

    def _reinitialize_camera(self):
        """
        Reinitializes the camera after multiple failures to read frames.
        """
        self.cap.release()
        if self.backend is not None:
            self.cap = cv2.VideoCapture(self.source, self.backend)
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        if self.cap.isOpened():
            print("Camera reinitialized successfully.")
            self.counter = 0
        else:
            print("Failed to reinitialize the camera.")

    def _reinitialize_camera(self):
        """
        Reinitializes the camera with exponential backoff after multiple failures.
        """
        self.cap.release()
        retry_attempts = 0
        while retry_attempts < 5:  # Retry up to 5 times
            print(f"Reinitializing camera (attempt {retry_attempts + 1})...")
            if self.backend is not None:
                self.cap = cv2.VideoCapture(self.source, self.backend)
            else:
                self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            if self.cap.isOpened():
                print("Camera reinitialized successfully.")
                self.counter = 0
                return
            else:
                retry_attempts += 1
                time.sleep(2 ** retry_attempts)  # Exponential backoff: 2, 4, 8, 16 seconds

        raise ValueError("Failed to reinitialize camera after multiple attempts.")

    def get_frame(self):
        """
        Retrieves the latest frame from the queue.

        :return: A numpy array representing the latest frame, or None if no frame is available.
        """
        if self.q.empty():
            print("Queue is empty. No frame to return.")
            return None
        try:
            return self.q.get_nowait()
        except queue.Empty:
            print("Queue unexpectedly empty during get_frame.")
            return None

    def release_camera(self):
        """
        Releases the camera resources and stops the thread.
        """
        self.stop_flag = True  # Signal the thread to stop
        self.t.join()  # Wait for the thread to finish
        self.cap.release()
        print("Camera released and thread stopped.")