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
from utils.species_names import is_known_species
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
        """Check if Telegram instant-alert notifications are enabled.

        Instant alerts fire when ``TELEGRAM_MODE`` is ``"live"`` (every
        detection above the cooldown) or ``"new_species_only"`` (only the
        very first time a species is ever seen). Legacy configs that set
        ``TELEGRAM_ENABLED`` alone (no mode key) are honoured so older
        deployments and tests keep working; the new mode switch, when
        present, always wins.
        """
        mode = self._mode
        if mode:
            return mode in ("live", "new_species_only")
        # Legacy path: no mode configured -> fall back to the old flag.
        return bool(self._config.get("TELEGRAM_ENABLED", False))

    @property
    def _mode(self) -> str:
        """Return the normalised TELEGRAM_MODE value (or empty string)."""
        return str(self._config.get("TELEGRAM_MODE", "") or "").strip().lower()

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

        Catalog-orphan species (e.g. classifier genus-fallbacks like
        ``Phoenicurus_sp.``) are dropped here so live alerts never send a
        Latin name the operator has no chance of recognising. This mirrors
        the gate applied to the daily / interval report path in
        ``utils.daily_report._fetch_species_best_photos``.

        In ``new_species_only`` mode the species is also checked against
        the persistent ``seen_species`` table; species already logged
        there are dropped silently so the operator only ever sees a
        first-of-its-kind alert.

        Args:
            species_info: Information about the detected species.
        """
        if not self.is_enabled:
            return

        locale = str(
            self._config.get("SPECIES_COMMON_NAME_LOCALE", "DE") or "DE"
        ).upper()
        if not is_known_species(species_info.latin_name, locale=locale):
            logger.debug(
                "Dropping catalog-orphan species from live alert: %r",
                species_info.latin_name,
            )
            return

        # New-species-only mode: skip if we've ever logged this species before.
        # The actual mark-as-seen INSERT happens after a successful send (in
        # send_summary) so a transport failure leaves the next attempt in
        # "still new" state.
        if self._mode == "new_species_only":
            from utils.db.seen_species import is_new_species

            if not is_new_species(species_info.latin_name):
                logger.debug(
                    "Suppressing already-seen species in new_species_only mode: %r",
                    species_info.latin_name,
                )
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

                device_name = str(
                    get_config().get("DEVICE_NAME", "") or ""
                ).strip()
                prefix = f"[{device_name}] " if device_name else ""

                # New-species-only mode gets its own headline so the operator
                # can tell a rarity ping apart from a routine live alert.
                if self._mode == "new_species_only":
                    headline = (
                        f"{prefix}✨ Neue Art entdeckt!"
                        if species_count == 1
                        else f"{prefix}✨ {species_count} neue Arten entdeckt!"
                    )
                    message = headline + "\n" + "\n".join(species_lines)
                else:
                    message = (
                        f"{prefix}🐦 {species_count} {art_text} detected:\n"
                        + "\n".join(species_lines)
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

                # In new_species_only mode, only mark species as seen AFTER a
                # successful send so a transport failure leaves the next
                # detection cycle in "still new" state and re-attempts.
                if self._mode == "new_species_only":
                    from utils.db.seen_species import mark_species_seen

                    for latin_name, info in species_to_send.items():
                        mark_species_seen(
                            latin_name,
                            image_filename=info.get("image_path"),
                            score=info.get("score"),
                        )

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
