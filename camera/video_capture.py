# ------------------------------------------------------------------------------
# VideoCapture Class for Threaded Video Capture
# ------------------------------------------------------------------------------

import cv2
import queue
import threading

class VideoCapture:
    """
    A class for threaded video capture, suitable for specialized applications requiring minimal latency.
    """

    _counter = 0

    def __init__(self, source=0, backend=None):
        """
        Initializes the video capture object with threading and queue.

        :param source: Camera source, such as an index for a connected camera (e.g., 0 for the default camera).
        """
        # Increment instance counter for multiple cameras
        VideoCapture._counter += 1
        self.num_instance = VideoCapture._counter

        self.backend = backend
        self.source = source

        # Initialize VideoCapture with optional backend
        if self.backend is not None:
            self.cap = cv2.VideoCapture(self.source, self.backend)
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)


        # Queue to store frames
        self.q = queue.Queue(maxsize=10)  # Limit to 10 frames
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.counter = 0
        self.t.start()

    def _reader(self):
        """
        Background thread to read frames continuously and keep only the latest frame.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.counter += 1
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            if self.counter == 10:  # Reinitialize camera after 10 dropped frames
                self.cap.release()

                # Initialize VideoCapture with optional backend
                if self.backend is not None:
                    self.cap = cv2.VideoCapture(self.source, self.backend)
                else:
                    self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

                if self.cap.isOpened():
                    self.counter = 0
                else:
                    print("Kamera konnte nach 10 Fehlern nicht neu initialisiert werden.")
                    break

    def get_frame(self):
        """
        Retrieves the latest frame from the queue.

        :return: A numpy array representing the latest frame, or None if no frame is available.
        """
        return self.q.get()

    def release_camera(self):
        """
        Releases the camera resources.
        """
        self.cap.release()