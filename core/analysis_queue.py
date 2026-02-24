import logging
import queue
import threading
import time
from collections.abc import Callable

from config import get_config

logger = logging.getLogger(__name__)


class AnalysisQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._processor_func: Callable | None = None

        # Reference to DetectionManager for gate control (injected via set_detection_manager)
        self._detection_manager = None

        # Secondary dedup: prevent same filename from being enqueued twice in one cycle
        self._pending_filenames: set[str] = set()
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def set_detection_manager(self, dm) -> None:
        """Inject DetectionManager for deep-scan gate control."""
        self._detection_manager = dm

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, processor_func: Callable):
        """
        Starts the background worker thread.
        processor_func: Function to call for each item. Must accept (item) as argument.
        """
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("AnalysisQueue worker already running")
            return

        self._processor_func = processor_func
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="DeepAnalysisWorker", daemon=True
        )
        self._worker_thread.start()
        logger.info("DeepAnalysisQueue worker started")

    def stop(self):
        """Stops the worker thread gracefully."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            logger.info("DeepAnalysisQueue worker stopped")

    # ------------------------------------------------------------------
    # Enqueue with secondary dedup (Constraint #4 / #5)
    # ------------------------------------------------------------------

    def enqueue(self, item) -> bool:
        """Adds an item to the queue.  Returns False if the filename is already pending."""
        filename = item.get("filename", "") if isinstance(item, dict) else ""
        if filename:
            with self._pending_lock:
                if filename in self._pending_filenames:
                    logger.debug(f"Dedup: {filename} already pending, skipping enqueue")
                    return False
                self._pending_filenames.add(filename)

        self._queue.put(item)
        logger.debug(f"Item enqueued. Queue size: {self._queue.qsize()}")
        return True

    # ------------------------------------------------------------------
    # Stats (Scope 3)
    # ------------------------------------------------------------------

    def pending_count(self) -> int:
        """Number of items waiting in the queue (approximate)."""
        return self._queue.qsize()

    def pending_filenames_count(self) -> int:
        """Number of distinct filenames tracked by the dedup set."""
        with self._pending_lock:
            return len(self._pending_filenames)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    @property
    def _gate_enabled(self) -> bool:
        return get_config().get("DEEP_SCAN_GATE_ENABLED", True)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                # Wait for an item, but check stop_event periodically
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            filename = item.get("filename", "") if isinstance(item, dict) else ""

            try:
                # --- Gate ON (clear = block loops) ---
                if self._gate_enabled and self._detection_manager:
                    self._detection_manager.enter_deep_scan_mode()
                    logger.info("Deep scan gate ON: pausing live DET+CLS")
                    time.sleep(1.0)  # Grace period for loops to park

                if self._processor_func:
                    logger.info(f"Processing analysis job: {item}")
                    self._processor_func(item)
            except Exception as e:
                logger.error(f"Error processing analysis job: {e}", exc_info=True)
            finally:
                # --- Gate OFF (set = unblock loops) — ALWAYS, even on exception ---
                if self._detection_manager:
                    self._detection_manager.exit_deep_scan_mode()
                    logger.info("Deep scan gate OFF: resuming live DET+CLS")

                # Remove filename from dedup set
                if filename:
                    with self._pending_lock:
                        self._pending_filenames.discard(filename)

                self._queue.task_done()


# Global instance
analysis_queue = AnalysisQueue()
