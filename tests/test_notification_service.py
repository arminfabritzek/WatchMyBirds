"""
Tests for NotificationService.

Tests the notification service in isolation without actual Telegram API calls.
"""

import threading
import time
from unittest.mock import patch

import pytest

from detectors.interfaces.notification import SpeciesInfo
from detectors.services.notification_service import NotificationService


@pytest.fixture
def common_names():
    """Sample common names mapping."""
    return {
        "Parus_major": "Kohlmeise",
        "Cyanistes_caeruleus": "Blaumeise",
        "Erithacus_rubecula": "Rotkehlchen",
    }


@pytest.fixture
def notification_service(common_names):
    """Create a notification service for testing."""
    with patch("detectors.services.notification_service.get_config") as mock_config:
        mock_config.return_value = {
            "TELEGRAM_ENABLED": True,
            "TELEGRAM_COOLDOWN": 60,
        }
        service = NotificationService(common_names=common_names)
        yield service


class TestNotificationServiceBasics:
    """Test basic functionality."""

    def test_is_enabled_when_configured(self, notification_service):
        """Service reports enabled when config says so."""
        assert notification_service.is_enabled is True

    def test_is_disabled_when_not_configured(self, common_names):
        """Service reports disabled when config says so."""
        with patch("detectors.services.notification_service.get_config") as mock_config:
            mock_config.return_value = {"TELEGRAM_ENABLED": False}
            service = NotificationService(common_names=common_names)
            assert service.is_enabled is False

    def test_pending_count_starts_at_zero(self, notification_service):
        """No pending detections initially."""
        assert notification_service.pending_count == 0

    def test_cooldown_from_config(self, notification_service):
        """Cooldown value comes from config."""
        assert notification_service.cooldown_seconds == 60


class TestQueueDetection:
    """Test detection queueing behavior."""

    def test_queue_single_detection(self, notification_service):
        """Can queue a single detection."""
        species = SpeciesInfo(
            latin_name="Parus_major",
            common_name="Kohlmeise",
            score=0.85,
            image_path="/path/to/image.jpg",
        )

        notification_service.queue_detection(species)

        assert notification_service.pending_count == 1

    def test_queue_multiple_species(self, notification_service):
        """Can queue multiple different species."""
        species1 = SpeciesInfo("Parus_major", "Kohlmeise", 0.85, "/img1.jpg")
        species2 = SpeciesInfo("Cyanistes_caeruleus", "Blaumeise", 0.90, "/img2.jpg")

        notification_service.queue_detection(species1)
        notification_service.queue_detection(species2)

        assert notification_service.pending_count == 2

    def test_higher_score_replaces_existing(self, notification_service):
        """Higher score detection replaces existing for same species."""
        low_score = SpeciesInfo("Parus_major", "Kohlmeise", 0.70, "/low.jpg")
        high_score = SpeciesInfo("Parus_major", "Kohlmeise", 0.95, "/high.jpg")

        notification_service.queue_detection(low_score)
        notification_service.queue_detection(high_score)

        # Still only 1 entry
        assert notification_service.pending_count == 1

    def test_lower_score_does_not_replace(self, notification_service):
        """Lower score detection does not replace existing."""
        high_score = SpeciesInfo("Parus_major", "Kohlmeise", 0.95, "/high.jpg")
        low_score = SpeciesInfo("Parus_major", "Kohlmeise", 0.70, "/low.jpg")

        notification_service.queue_detection(high_score)
        notification_service.queue_detection(low_score)

        assert notification_service.pending_count == 1

    def test_queue_when_disabled_is_noop(self, common_names):
        """Queueing when disabled does nothing."""
        with patch("detectors.services.notification_service.get_config") as mock_config:
            mock_config.return_value = {"TELEGRAM_ENABLED": False}
            service = NotificationService(common_names=common_names)

            species = SpeciesInfo("Parus_major", "Kohlmeise", 0.85, "/img.jpg")
            service.queue_detection(species)

            assert service.pending_count == 0


class TestShouldSend:
    """Test should_send logic."""

    def test_should_not_send_when_disabled(self, common_names):
        """Should not send when notifications are disabled."""
        with patch("detectors.services.notification_service.get_config") as mock_config:
            mock_config.return_value = {"TELEGRAM_ENABLED": False}
            service = NotificationService(common_names=common_names)

            assert service.should_send() is False

    def test_should_not_send_when_empty(self, notification_service):
        """Should not send when no pending detections."""
        # Force cooldown expired
        notification_service._last_notification_time = 0

        assert notification_service.should_send() is False

    def test_should_not_send_during_cooldown(self, notification_service):
        """Should not send during cooldown period."""
        species = SpeciesInfo("Parus_major", "Kohlmeise", 0.85, "/img.jpg")
        notification_service.queue_detection(species)

        # Set recent notification
        notification_service._last_notification_time = time.time()

        assert notification_service.should_send() is False

    def test_should_send_after_cooldown(self, notification_service):
        """Should send after cooldown with pending detections."""
        species = SpeciesInfo("Parus_major", "Kohlmeise", 0.85, "/img.jpg")
        notification_service.queue_detection(species)

        # Set old notification time
        notification_service._last_notification_time = time.time() - 100

        assert notification_service.should_send() is True


class TestSendSummary:
    """Test summary sending."""

    @patch("detectors.services.notification_service.send_telegram_message")
    def test_send_summary_success(self, mock_send, notification_service):
        """Successfully sends summary message."""
        mock_send.return_value = [{"ok": True}]

        species = SpeciesInfo("Parus_major", "Kohlmeise", 0.85, "/img.jpg")
        notification_service.queue_detection(species)
        notification_service._last_notification_time = 0

        result = notification_service.send_summary()

        assert result is True
        assert mock_send.called
        assert notification_service.pending_count == 0

    @patch("detectors.services.notification_service.send_telegram_message")
    def test_send_summary_clears_pending(self, mock_send, notification_service):
        """Sending clears pending queue."""
        mock_send.return_value = [{"ok": True}]

        notification_service.queue_detection(
            SpeciesInfo("Parus_major", "K", 0.8, "/a.jpg")
        )
        notification_service.queue_detection(
            SpeciesInfo("Cyanistes_caeruleus", "B", 0.9, "/b.jpg")
        )
        notification_service._last_notification_time = 0

        notification_service.send_summary()

        assert notification_service.pending_count == 0

    @patch("detectors.services.notification_service.send_telegram_message")
    def test_send_summary_formats_message(self, mock_send, notification_service):
        """Message includes all species in correct format."""
        mock_send.return_value = [{"ok": True}]

        notification_service.queue_detection(
            SpeciesInfo("Parus_major", "Kohlmeise", 0.8, "/a.jpg")
        )
        notification_service.queue_detection(
            SpeciesInfo("Cyanistes_caeruleus", "Blaumeise", 0.9, "/b.jpg")
        )
        notification_service._last_notification_time = 0

        notification_service.send_summary()

        call_args = mock_send.call_args
        message = call_args.kwargs["text"]

        assert "2 Species detected" in message
        assert "Kohlmeise" in message
        assert "Blaumeise" in message
        assert "Parus major" in message  # Underscore replaced with space

    @patch("detectors.services.notification_service.send_telegram_message")
    def test_uses_best_image(self, mock_send, notification_service):
        """Uses image from highest scoring species."""
        mock_send.return_value = [{"ok": True}]

        notification_service.queue_detection(SpeciesInfo("Low", "Low", 0.5, "/low.jpg"))
        notification_service.queue_detection(
            SpeciesInfo("High", "High", 0.9, "/high.jpg")
        )
        notification_service._last_notification_time = 0

        notification_service.send_summary()

        call_args = mock_send.call_args
        assert call_args.kwargs["photo_path"] == "/high.jpg"


class TestClearAndReset:
    """Test clear and reset operations."""

    def test_clear_queue(self, notification_service):
        """Clear removes all pending detections."""
        notification_service.queue_detection(SpeciesInfo("A", "A", 0.8, "/a.jpg"))
        notification_service.queue_detection(SpeciesInfo("B", "B", 0.9, "/b.jpg"))

        notification_service.clear_queue()

        assert notification_service.pending_count == 0

    def test_reset_cooldown(self, notification_service):
        """Reset updates last notification time."""
        old_time = notification_service._last_notification_time

        notification_service.reset_cooldown()

        assert notification_service._last_notification_time > old_time


class TestCommonNameResolution:
    """Test common name lookup."""

    def test_get_known_common_name(self, notification_service):
        """Returns common name for known species."""
        assert notification_service.get_common_name("Parus_major") == "Kohlmeise"

    def test_get_unknown_formats_latin(self, notification_service):
        """Returns formatted Latin name for unknown species."""
        result = notification_service.get_common_name("Unknown_species")
        assert result == "Unknown species"

    def test_create_species_info_factory(self, notification_service):
        """Factory creates SpeciesInfo with resolved common name."""
        info = notification_service.create_species_info(
            latin_name="Parus_major", score=0.85, image_path="/test.jpg"
        )

        assert info.latin_name == "Parus_major"
        assert info.common_name == "Kohlmeise"
        assert info.score == 0.85
        assert info.image_path == "/test.jpg"


class TestThreadSafety:
    """Test thread-safety of operations."""

    def test_concurrent_queue_operations(self, notification_service):
        """Multiple threads can queue concurrently without issues."""

        def worker(species_prefix: str, count: int):
            for i in range(count):
                info = SpeciesInfo(
                    f"{species_prefix}_{i}",
                    f"Common_{i}",
                    0.5 + (i / 100),
                    f"/img_{i}.jpg",
                )
                notification_service.queue_detection(info)

        threads = [
            threading.Thread(target=worker, args=("Thread1", 50)),
            threading.Thread(target=worker, args=("Thread2", 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 100 unique species
        assert notification_service.pending_count == 100
