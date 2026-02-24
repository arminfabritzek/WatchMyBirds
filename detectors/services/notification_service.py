"""
Notification Service - Telegram Detection Notifications.

Implements NotificationInterface for sending detection alerts via Telegram.
Extracts notification logic from DetectionManager for independent operation.
"""

import threading
import time

from config import get_config
from detectors.interfaces.notification import NotificationInterface, SpeciesInfo
from logging_config import get_logger
from utils.telegram_notifier import send_telegram_message

logger = get_logger(__name__)


class NotificationService(NotificationInterface):
    """
    Handles Telegram notifications for bird detections.

    Features:
    - Collects detections during cooldown period
    - Sends summary with all observed species when cooldown expires
    - Thread-safe for concurrent detection processing
    - Keeps best image per species (highest score)
    """

    def __init__(self, common_names: dict | None = None):
        """
        Initialize the notification service.

        Args:
            common_names: Optional dict mapping Latin names to common names.
                         If not provided, uses Latin name formatted as common.
        """
        self._config = get_config()
        self._common_names = common_names or {}

        # State for detection collection
        self._pending_species: dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        # Cooldown tracking
        self._send_lock = threading.Lock()
        self._last_notification_time = 0.0

    @property
    def is_enabled(self) -> bool:
        """Check if Telegram notifications are enabled."""
        return self._config.get("TELEGRAM_ENABLED", False)

    @property
    def pending_count(self) -> int:
        """Return number of pending species detections."""
        with self._pending_lock:
            return len(self._pending_species)

    @property
    def cooldown_seconds(self) -> int:
        """Get the configured cooldown period in seconds."""
        return self._config.get("TELEGRAM_COOLDOWN", 60)

    def queue_detection(self, species_info: SpeciesInfo) -> None:
        """
        Queue a detection for notification.

        Only updates the entry if the new score is higher than existing.
        This ensures we send the best image for each species.

        Args:
            species_info: Information about the detected species.
        """
        if not self.is_enabled:
            return

        with self._pending_lock:
            existing = self._pending_species.get(species_info.latin_name)

            # Only update if this is a new species or has higher score
            if existing is None or species_info.score > existing["score"]:
                self._pending_species[species_info.latin_name] = {
                    "common": species_info.common_name,
                    "score": species_info.score,
                    "image_path": species_info.image_path,
                }
                logger.debug(
                    f"Queued detection: {species_info.latin_name} "
                    f"({species_info.common_name}) score={species_info.score:.2f}"
                )

    def should_send(self) -> bool:
        """
        Check if cooldown has expired and we should send.

        Returns:
            True if cooldown expired and there are pending detections.
        """
        if not self.is_enabled:
            return False

        current_time = time.time()
        cooldown_expired = (
            current_time - self._last_notification_time
        ) >= self.cooldown_seconds

        with self._pending_lock:
            has_pending = len(self._pending_species) > 0

        return cooldown_expired and has_pending

    def send_summary(self) -> bool:
        """
        Send a summary notification of all queued detections.

        Thread-safe with double-checked locking to prevent duplicate sends.

        Returns:
            True if notification was sent successfully.
        """
        if not self.is_enabled:
            return False

        current_time = time.time()

        # Fast path check without lock
        if (current_time - self._last_notification_time) < self.cooldown_seconds:
            return False

        with self._send_lock:
            # Double-check after acquiring lock
            if (current_time - self._last_notification_time) < self.cooldown_seconds:
                return False

            with self._pending_lock:
                if not self._pending_species:
                    return False

                # Build summary message
                species_count = len(self._pending_species)

                # Sort by score descending
                sorted_species = sorted(
                    self._pending_species.items(),
                    key=lambda x: x[1]["score"],
                    reverse=True,
                )

                # Format message: "🐦 X Species detected:" + list
                art_text = "Species"
                species_lines = []

                for latin_name, info in sorted_species:
                    latin_formatted = latin_name.replace("_", " ")
                    species_lines.append(f"• {info['common']} ({latin_formatted})")

                message = f"🐦 {species_count} {art_text} detected:\n" + "\n".join(
                    species_lines
                )

                # Use image of highest scoring species
                image_path = sorted_species[0][1]["image_path"]

                # Copy data before clearing
                species_to_send = dict(self._pending_species)
                self._pending_species.clear()

            # Send outside the pending lock (network I/O)
            try:
                send_telegram_message(text=message, photo_path=image_path)
                self._last_notification_time = current_time

                logger.info(
                    f"Telegram notification sent: {len(species_to_send)} species, "
                    f"best: {sorted_species[0][0]}"
                )
                return True

            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")
                # Re-queue the detections on failure
                with self._pending_lock:
                    for latin_name, info in species_to_send.items():
                        if latin_name not in self._pending_species:
                            self._pending_species[latin_name] = info
                return False

    def reset_cooldown(self) -> None:
        """Reset the cooldown timer to now."""
        with self._send_lock:
            self._last_notification_time = time.time()

    def clear_queue(self) -> None:
        """Clear all pending detections."""
        with self._pending_lock:
            self._pending_species.clear()

    def get_common_name(self, latin_name: str) -> str:
        """
        Get common name for a species.

        Args:
            latin_name: Latin species name.

        Returns:
            Common name if available, otherwise formatted Latin name.
        """
        return self._common_names.get(latin_name, latin_name.replace("_", " "))

    def create_species_info(
        self, latin_name: str, score: float, image_path: str
    ) -> SpeciesInfo:
        """
        Factory method to create SpeciesInfo with auto-resolved common name.

        Args:
            latin_name: Latin species name.
            score: Detection/classification score.
            image_path: Path to the image for notification.

        Returns:
            SpeciesInfo with resolved common name.
        """
        return SpeciesInfo(
            latin_name=latin_name,
            common_name=self.get_common_name(latin_name),
            score=score,
            image_path=image_path,
        )
