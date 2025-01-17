# ------------------------------------------------------------------------------
# BaseCamera Class for Capturing Video Frames
# ------------------------------------------------------------------------------

import cv2

class BaseCamera:
    """
    A class to handle video capture from a camera source.
    """

    def __init__(self, source=0):
        """
        Initializes the camera source.

        :param source: Camera source, such as an index for a connected camera
                       (e.g., 0 for the default camera) or a URL for an IP camera.
        """
        self.source = source
        self.capture = cv2.VideoCapture(source)

    def get_frame(self):
        """
        Reads a single frame from the camera.

        :return: A numpy array representing the current frame, or None if no frame is available.
        """
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None

    def release(self):
        """
        Releases the camera resources.
        """
        self.capture.release()