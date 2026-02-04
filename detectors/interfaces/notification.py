"""
Notification Interface - Telegram Notifications.

Defines the contract for sending detection notifications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SpeciesInfo:
    """
    Information about a detected species for notification.

    Attributes:
        latin_name: Latin species name (e.g., "Parus_major").
        common_name: Common name in configured language (e.g., "Kohlmeise").
        score: Combined detection+classification score.
        image_path: Path to the image to include in notification.
    """

    latin_name: str
    common_name: str
    score: float
    image_path: str


class NotificationInterface(ABC):
    """
    Interface for detection notifications.

    Implementations should handle:
    - Collecting detections during cooldown period
    - Sending summary notifications
    - Rate limiting / cooldown enforcement
    """

    @abstractmethod
    def queue_detection(self, species_info: SpeciesInfo) -> None:
        """
        Queues a detection for notification.

        Multiple detections during cooldown period are collected
        and sent as a summary when cooldown expires.

        Args:
            species_info: Information about the detected species.
        """
        pass

    @abstractmethod
    def should_send(self) -> bool:
        """
        Checks if the cooldown period has expired.

        Returns:
            True if enough time has passed since last notification.
        """
        pass

    @abstractmethod
    def send_summary(self) -> bool:
        """
        Sends a summary notification of all queued detections.

        Returns:
            True if notification was sent successfully.
        """
        pass

    @abstractmethod
    def reset_cooldown(self) -> None:
        """
        Resets the cooldown timer.

        Should be called after sending a notification.
        """
        pass

    @abstractmethod
    def clear_queue(self) -> None:
        """
        Clears the pending detection queue.

        Should be called after sending a notification.
        """
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Checks if notifications are enabled.

        Returns:
            True if notifications are configured and enabled.
        """
        pass

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """
        Returns the number of pending detections.

        Returns:
            Number of species in the queue.
        """
        pass
