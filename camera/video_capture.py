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

    def __init__(self, source=0):
        """
        Initializes the video capture object with threading and queue.

        :param source: Camera source, such as an index for a connected camera (e.g., 0 for the default camera).
        """
        # Increment instance counter for multiple cameras
        VideoCapture._counter += 1
        self.num_instance = VideoCapture._counter

        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        # Queue to store frames
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()
        self.counter = 0

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
                self.cap = cv2.VideoCapture(self.source)

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