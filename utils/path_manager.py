from pathlib import Path


# Directory structure:
# data/
# ├── originals/
# │   └── YYYY-MM-DD/
# │       └── filename
# └── derivatives/
#     ├── thumbs/
#     │   └── YYYY-MM-DD/
#     │       └── filename
#     └── optimized/
#         └── YYYY-MM-DD/
#             └── filename


class PathManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.originals_dir = self.base_dir / "originals"
        self.derivatives_dir = self.base_dir / "derivatives"
        self.thumbs_dir = self.derivatives_dir / "thumbs"
        self.optimized_dir = self.derivatives_dir / "optimized"
        # Inbox directories (for web upload ingest)
        self.inbox_dir = self.base_dir / "inbox"
        self.inbox_pending_dir = self.inbox_dir / "pending"
        self.inbox_error_dir = self.inbox_dir / "error"
        # Backup directory
        self.backup_dir = self.base_dir / "backup"

    # -------------------------------------------------------------------------
    # Inbox Path Methods
    # -------------------------------------------------------------------------
    def get_inbox_root_dir(self) -> Path:
        """Returns the inbox root directory."""
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        return self.inbox_dir

    def get_inbox_pending_dir(self) -> Path:
        """Returns the inbox/pending directory, creates if needed."""
        self.inbox_pending_dir.mkdir(parents=True, exist_ok=True)
        return self.inbox_pending_dir

    def get_inbox_processed_dir(self, date_str: str) -> Path:
        """
        Returns inbox/processed/YYYYMMDD directory.
        date_str should be in YYYYMMDD format.
        """
        processed_dir = self.inbox_dir / "processed" / date_str
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir

    def get_inbox_skipped_dir(self, date_str: str) -> Path:
        """
        Returns inbox/skipped/YYYYMMDD directory.
        date_str should be in YYYYMMDD format.
        """
        skipped_dir = self.inbox_dir / "skipped" / date_str
        skipped_dir.mkdir(parents=True, exist_ok=True)
        return skipped_dir

    def get_inbox_error_dir(self) -> Path:
        """Returns the inbox/error directory, creates if needed."""
        self.inbox_error_dir.mkdir(parents=True, exist_ok=True)
        return self.inbox_error_dir

    # -------------------------------------------------------------------------
    # Backup Path Methods
    # -------------------------------------------------------------------------
    def get_backup_dir(self) -> Path:
        """Returns the backup directory, creates if needed."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        return self.backup_dir

    def get_backup_tmp_db_path(self) -> Path:
        """
        Returns a unique temp DB path for backup operations.
        Format: backup/tmp_db_YYYYMMDD_HHMMSS.db
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.get_backup_dir() / f"tmp_db_{timestamp}.db"

    def get_date_folder(self, date_str: str) -> str:
        """Returns the YYYY-MM-DD folder name from various inputs."""
        # Assume input is either YYYY-MM-DD or YYYYMMDD prefix
        if "-" not in date_str and len(date_str) >= 8:
            # Convert YYYYMMDD... to YYYY-MM-DD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def ensure_date_structure(self, date_str: str):
        """Creates the necessary folder structure for a given date."""
        date_folder = self.get_date_folder(date_str)
        for folder in [self.originals_dir, self.thumbs_dir, self.optimized_dir]:
            (folder / date_folder).mkdir(parents=True, exist_ok=True)

    def get_original_path(self, filename: str) -> Path:
        """Resolves the absolute path for an original image."""
        date_str = self.extract_date_from_filename(filename)
        date_folder = self.get_date_folder(date_str)
        return self.originals_dir / date_folder / filename

    def get_derivative_path(self, filename: str, type: str = "thumb") -> Path:
        """
        Resolves the absolute path for a derivative.
        type: 'thumb' | 'optimized'
        """
        date_str = self.extract_date_from_filename(filename)
        date_folder = self.get_date_folder(date_str)
        file_stem = Path(filename).stem
        derivative_name = f"{file_stem}.webp"

        if type == "thumb":
            return self.thumbs_dir / date_folder / derivative_name
        elif type == "optimized":
            return self.optimized_dir / date_folder / derivative_name
        else:
            raise ValueError(f"Unknown derivative type: {type}")

    def get_preview_thumb_path(self, filename: str) -> Path:
        """
        Resolves the absolute path for a preview thumbnail.
        Used for orphan images without detection-based crops.
        Preview thumbs are stored alongside detection thumbs in thumbs/ directory.
        """
        date_str = self.extract_date_from_filename(filename)
        date_folder = self.get_date_folder(date_str)
        file_stem = Path(filename).stem
        preview_name = f"{file_stem}_preview.webp"
        return self.thumbs_dir / date_folder / preview_name

    def extract_date_from_filename(self, filename: str) -> str:
        """
        Extracts date from standard filename format: YYYYMMDD_HHMMSS_...
        Returns YYYY-MM-DD string.
        """
        parts = filename.split("_")
        if len(parts) > 0 and len(parts[0]) == 8 and parts[0].isdigit():
            ds = parts[0]
            return f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        # Fallback or error - application should ensure valid filenames
        # For robustness, try to parse
        return "unknown_date"


# Global Instance - to be initialized by app with config["OUTPUT_DIR"]
_instance = None


def get_path_manager(output_dir: str = None) -> PathManager:
    global _instance
    if _instance is None:
        if output_dir is None:
            # Default fallback if called before init
            from config import get_config

            output_dir = get_config()["OUTPUT_DIR"]
        _instance = PathManager(output_dir)
    return _instance
