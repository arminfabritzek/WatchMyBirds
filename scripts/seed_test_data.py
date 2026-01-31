#!/usr/bin/env python3
"""
Seed Script for WatchMyBirds Test Data

Creates reproducible test data for UI testing:
- SQLite database with test entries
- Placeholder images in output directories
- Various review states (untagged, confirmed, rejected, no_bird)
- Multiple species with different confidence levels

Usage:
    python scripts/seed_test_data.py              # Fresh seed (clears existing)
    python scripts/seed_test_data.py --append     # Append to existing data
    python scripts/seed_test_data.py --dry-run    # Show what would be created

Requirements:
    - Pillow (for image generation)
    - Run from project root directory

Output:
    - output/images.db (SQLite database)
    - output/originals/YYYY-MM-DD/*.jpg
    - output/derivatives/thumbs/YYYY-MM-DD/*.webp
    - output/derivatives/optimized/YYYY-MM-DD/*.webp
"""

import argparse
import hashlib
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Test data configuration
TEST_CONFIG = {
    "output_dir": "output",
    "days_back": 5,           # Generate data for last N days
    "images_per_day": 10,     # Images per day
    "detections_per_image": (1, 3),  # Min/max detections per image
    
    # Species distribution (name: weight)
    "species": {
        "Amsel": 0.25,
        "Blaumeise": 0.20,
        "Kohlmeise": 0.15,
        "Rotkehlchen": 0.12,
        "Spatz": 0.10,
        "Buchfink": 0.08,
        "Grünfink": 0.05,
        "Elster": 0.03,
        "Star": 0.02,
    },
    
    # Review status distribution
    "review_states": {
        "untagged": 0.60,
        "confirmed_bird": 0.30,
        "no_bird": 0.08,
        "rejected": 0.02,  # For detections
    },
    
    # Confidence ranges
    "high_confidence": (0.85, 0.99),
    "medium_confidence": (0.65, 0.85),
    "low_confidence": (0.35, 0.65),
}

# Color palette for placeholder images
SPECIES_COLORS = {
    "Amsel": (40, 40, 40),        # Dark grey/black
    "Blaumeise": (70, 130, 180),  # Steel blue
    "Kohlmeise": (255, 215, 0),   # Gold
    "Rotkehlchen": (205, 92, 92), # Indian red
    "Spatz": (139, 119, 101),     # Brown
    "Buchfink": (255, 182, 193),  # Light pink
    "Grünfink": (85, 107, 47),    # Dark olive green
    "Elster": (0, 0, 0),          # Black
    "Star": (72, 61, 139),        # Dark slate blue
}


# =============================================================================
# Database Functions
# =============================================================================

def init_database(db_path: Path, clear: bool = True) -> sqlite3.Connection:
    """Initialize or reset the test database."""
    if clear and db_path.exists():
        db_path.unlink()
        print(f"  Cleared existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    
    # Create schema (simplified from utils/db.py)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            uri TEXT,
            config_json TEXT,
            active INTEGER DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            coco_json TEXT,
            downloaded_timestamp TEXT,
            detector_model_id TEXT,
            classifier_model_id TEXT,
            source_id INTEGER REFERENCES sources(source_id),
            content_hash TEXT,
            review_status TEXT DEFAULT 'untagged',
            review_updated_at TEXT,
            max_detection_confidence REAL
        );
        
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            od_model_id TEXT,
            created_at TEXT,
            score REAL,
            status TEXT DEFAULT 'active',
            thumbnail_path TEXT,
            FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            cls_model_id TEXT,
            rank INTEGER DEFAULT 1,
            created_at TEXT,
            status TEXT DEFAULT 'active',
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_detections_filename ON detections(image_filename);
        CREATE INDEX IF NOT EXISTS idx_detections_status ON detections(status);
        CREATE INDEX IF NOT EXISTS idx_classifications_detection ON classifications(detection_id);
        CREATE INDEX IF NOT EXISTS idx_images_review_status ON images(review_status);
    """)
    
    # Create default source
    conn.execute("""
        INSERT OR IGNORE INTO sources (source_id, name, type)
        VALUES (1, 'Test Seed', 'test_data')
    """)
    conn.commit()
    
    return conn


def insert_image(conn: sqlite3.Connection, filename: str, timestamp: str,
                 content_hash: str, review_status: str = "untagged") -> None:
    """Insert an image record."""
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO images (filename, timestamp, source_id, content_hash, review_status, review_updated_at)
        VALUES (?, ?, 1, ?, ?, ?)
    """, (filename, timestamp, content_hash, review_status, now))


def insert_detection(conn: sqlite3.Connection, image_filename: str, 
                     bbox: tuple, od_class: str, od_conf: float,
                     score: float, status: str = "active",
                     thumbnail_path: str = None) -> int:
    """Insert a detection record and return its ID."""
    now = datetime.now().isoformat()
    cursor = conn.execute("""
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, od_confidence, score, status, created_at, thumbnail_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_filename, bbox[0], bbox[1], bbox[2], bbox[3],
          od_class, od_conf, score, status, now, thumbnail_path))
    return cursor.lastrowid


def insert_classification(conn: sqlite3.Connection, detection_id: int,
                          cls_class: str, cls_conf: float) -> int:
    """Insert a classification record."""
    now = datetime.now().isoformat()
    cursor = conn.execute("""
        INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, created_at)
        VALUES (?, ?, ?, 1, ?)
    """, (detection_id, cls_class, cls_conf, now))
    return cursor.lastrowid


# =============================================================================
# Image Generation Functions
# =============================================================================

def generate_placeholder_image(path: Path, width: int, height: int,
                               species: str = None, timestamp: str = None,
                               detection_index: int = 1) -> str:
    """Generate a placeholder image with species-colored rectangle and text."""
    # Create base image with gradient background
    img = Image.new("RGB", (width, height), (200, 220, 200))
    draw = ImageDraw.Draw(img)
    
    # Add gradient
    for y in range(height):
        shade = int(180 + (y / height) * 40)
        for x in range(width):
            img.putpixel((x, y), (shade - 20, shade, shade - 10))
    
    # Add species-colored box
    if species:
        color = SPECIES_COLORS.get(species, (100, 100, 100))
        box_size = min(width, height) // 3
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255), width=3)
    
    # Add text label
    label = f"{species or 'Bird'} #{detection_index}"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    
    # Get text bounding box
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height - 40
    
    # Draw text with background
    draw.rectangle([text_x - 5, text_y - 5, text_x + text_width + 5, text_y + 25],
                   fill=(255, 255, 255, 200))
    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
    
    # Add timestamp
    if timestamp:
        ts_label = timestamp[9:17]  # HHMMSS part
        draw.text((10, 10), ts_label, fill=(100, 100, 100), font=font)
    
    # Save image
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=85)
    
    # Return content hash
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_thumbnail(original_path: Path, thumb_path: Path, size: int = 256) -> None:
    """Generate a thumbnail from the original image."""
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Image.open(original_path) as img:
        # Center crop to square
        min_dim = min(img.size)
        left = (img.width - min_dim) // 2
        top = (img.height - min_dim) // 2
        cropped = img.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize
        resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(thumb_path, "WEBP", quality=80)


def generate_optimized(original_path: Path, optimized_path: Path, 
                       max_width: int = 1920) -> None:
    """Generate an optimized version of the image."""
    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Image.open(original_path) as img:
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        img.save(optimized_path, "WEBP", quality=85)


# =============================================================================
# Data Generation
# =============================================================================

def weighted_choice(options: dict) -> str:
    """Select from weighted options."""
    choices = list(options.keys())
    weights = list(options.values())
    return random.choices(choices, weights=weights, k=1)[0]


def generate_random_bbox() -> tuple:
    """Generate random bounding box (x, y, w, h) in normalized coords."""
    x = random.uniform(0.1, 0.6)
    y = random.uniform(0.1, 0.6)
    w = random.uniform(0.15, 0.35)
    h = random.uniform(0.15, 0.35)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def generate_test_data(output_dir: Path, dry_run: bool = False) -> dict:
    """Generate complete test dataset."""
    stats = {
        "images": 0,
        "detections": 0,
        "classifications": 0,
        "files_created": 0,
    }
    
    cfg = TEST_CONFIG
    
    # Initialize database
    db_path = output_dir / "images.db"
    if not dry_run:
        conn = init_database(db_path, clear=True)
    
    # Generate data for each day
    today = datetime.now()
    
    for day_offset in range(cfg["days_back"]):
        current_date = today - timedelta(days=day_offset)
        date_str = current_date.strftime("%Y%m%d")
        date_folder = current_date.strftime("%Y-%m-%d")
        
        print(f"\n📅 Generating data for {date_folder}...")
        
        for img_idx in range(cfg["images_per_day"]):
            # Generate random time
            hour = random.randint(6, 19)  # Daylight hours
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            time_str = f"{hour:02d}{minute:02d}{second:02d}"
            
            # Generate filename
            timestamp = f"{date_str}_{time_str}"
            unique_id = random.randint(100000, 999999)
            filename = f"{timestamp}_{unique_id}.jpg"
            
            # Determine review status for this image
            review_status = weighted_choice(cfg["review_states"])
            
            # Skip creating actual items for 'rejected' (applied to detections, not images)
            if review_status == "rejected":
                review_status = "untagged"
            
            # Paths
            original_path = output_dir / "originals" / date_folder / filename
            
            # Determine primary species
            primary_species = weighted_choice(cfg["species"])
            
            # Generate placeholder image
            if not dry_run:
                content_hash = generate_placeholder_image(
                    original_path, 1920, 1080, 
                    species=primary_species, 
                    timestamp=timestamp
                )
                stats["files_created"] += 1
                
                # Insert image record
                insert_image(conn, filename, timestamp, content_hash, review_status)
            
            stats["images"] += 1
            
            # Generate detections (skip for no_bird images)
            if review_status != "no_bird":
                num_detections = random.randint(*cfg["detections_per_image"])
                max_conf = 0.0
                
                for det_idx in range(num_detections):
                    # Generate detection data
                    bbox = generate_random_bbox()
                    species = primary_species if det_idx == 0 else weighted_choice(cfg["species"])
                    
                    # Confidence (vary across range)
                    conf_type = random.choice(["high", "medium", "low"])
                    if conf_type == "high":
                        od_conf = random.uniform(*cfg["high_confidence"])
                        cls_conf = random.uniform(*cfg["high_confidence"])
                    elif conf_type == "medium":
                        od_conf = random.uniform(*cfg["medium_confidence"])
                        cls_conf = random.uniform(*cfg["medium_confidence"])
                    else:
                        od_conf = random.uniform(*cfg["low_confidence"])
                        cls_conf = random.uniform(*cfg["low_confidence"])
                    
                    score = (od_conf + cls_conf) / 2
                    max_conf = max(max_conf, score)
                    
                    # Determine detection status
                    det_status = "active"
                    if random.random() < 0.05:  # 5% rejected
                        det_status = "rejected"
                    
                    # Thumbnail filename
                    thumb_filename = filename.replace(".jpg", f"_crop_{det_idx + 1}.webp")
                    thumb_path = output_dir / "derivatives" / "thumbs" / date_folder / thumb_filename
                    
                    if not dry_run:
                        # Insert detection
                        detection_id = insert_detection(
                            conn, filename, bbox, "bird", od_conf, score, 
                            det_status, thumb_filename
                        )
                        
                        # Insert classification
                        insert_classification(conn, detection_id, species, cls_conf)
                        
                        # Generate thumbnail
                        generate_thumbnail(original_path, thumb_path)
                        stats["files_created"] += 1
                    
                    stats["detections"] += 1
                    stats["classifications"] += 1
                
                # Generate optimized image
                opt_filename = filename.replace(".jpg", ".webp")
                opt_path = output_dir / "derivatives" / "optimized" / date_folder / opt_filename
                
                if not dry_run:
                    generate_optimized(original_path, opt_path)
                    stats["files_created"] += 1
                    
                    # Update max detection confidence on image
                    conn.execute(
                        "UPDATE images SET max_detection_confidence = ? WHERE filename = ?",
                        (round(max_conf, 4), filename)
                    )
    
    if not dry_run:
        conn.commit()
        conn.close()
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for WatchMyBirds UI testing"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be created without actually creating files"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=5,
        help="Number of days to generate (default: 5)"
    )
    parser.add_argument(
        "--images-per-day", "-i",
        type=int,
        default=10,
        help="Images per day (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Update config
    TEST_CONFIG["days_back"] = args.days
    TEST_CONFIG["images_per_day"] = args.images_per_day
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("🐦 WatchMyBirds Test Data Seed Script")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Days to generate: {args.days}")
    print(f"Images per day: {args.images_per_day}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files will be created\n")
    
    stats = generate_test_data(output_dir, dry_run=args.dry_run)
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"  Images:          {stats['images']}")
    print(f"  Detections:      {stats['detections']}")
    print(f"  Classifications: {stats['classifications']}")
    if not args.dry_run:
        print(f"  Files created:   {stats['files_created']}")
        print(f"\n✅ Test data generated successfully!")
        print(f"   Database: {output_dir / 'images.db'}")
    else:
        print(f"\n📋 Would create {stats['files_created']} files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
