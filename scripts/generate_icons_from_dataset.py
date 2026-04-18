#!/usr/bin/env python3
"""
generate_icons_from_dataset.py
==============================
Generate bold flat-art bird icons for every species in a YOLO training
dataset, fully automatic - no manual per-species colour coding.

Runs on a user-supplied dataset folder and produces one PNG per class.

Usage
-----
    python generate_icons_from_dataset.py DATASET_DIR
        [--out OUT_DIR]          default: DATASET_DIR/_icons/
        [--samples N]            default: 40
        [--clusters K]           default: 6
        [--force]                rebuild colour cache
        [--only Class1,Class2]   restrict to specific classes
        [--workers N]            parallel workers (default: os.cpu_count())

Supported dataset layouts
--------------------------
Layout A (standard YOLOv5/v8):
    dataset/
      images/train/*.jpg   (and/or val/, test/)
      labels/train/*.txt
      data.yaml  OR  classes.txt

Layout B (flat):
    dataset/
      *.jpg
      *.txt
      classes.txt

Dependencies
------------
Required : Pillow, pycairo, pyyaml
Preferred: numpy, scikit-learn  (fast k-means in Lab space)
Fallback : pure-Python k-means  (slower, no extra deps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy deps — detect at import time
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from sklearn.cluster import KMeans as _SKLearnKMeans  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from skimage.color import rgb2lab as _skimage_rgb2lab  # noqa: F401
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.WARNING,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cairo / template imports (same package path trick as the vector script)
# ---------------------------------------------------------------------------
_CAIRO_SITE = "/opt/homebrew/lib/python3.14/site-packages"
if _CAIRO_SITE not in sys.path:
    sys.path.insert(0, _CAIRO_SITE)

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_species_icon_vector import (  # noqa: E402
    draw_cuckoo,
    draw_corvid,
    draw_finch,
    draw_nuthatch_creeper,
    draw_pigeon,
    draw_swift_swallow,
    draw_thrush,
    draw_tit,
    draw_wagtail_starling,
    draw_warbler,
    draw_woodpecker,
    new_surface,
)

# ---------------------------------------------------------------------------
# Genus -> template mapping
# ---------------------------------------------------------------------------
GENUS_TEMPLATES: dict[str, Any] = {
    # Tits (Paridae + Aegithalidae)
    "Parus": draw_tit,
    "Cyanistes": draw_tit,
    "Periparus": draw_tit,
    "Poecile": draw_tit,
    "Lophophanes": draw_tit,
    "Aegithalos": draw_tit,
    "Baeolophus": draw_tit,
    "Melaniparus": draw_tit,
    # Finches / buntings / sparrows / accentors
    "Fringilla": draw_finch,
    "Chloris": draw_finch,
    "Carduelis": draw_finch,
    "Spinus": draw_finch,
    "Pyrrhula": draw_finch,
    "Emberiza": draw_finch,
    "Passer": draw_finch,
    "Prunella": draw_finch,
    "Coccothraustes": draw_finch,
    "Serinus": draw_finch,
    "Linaria": draw_finch,
    "Acanthis": draw_finch,
    "Loxia": draw_finch,
    "Bucanetes": draw_finch,
    "Carpodacus": draw_finch,
    "Haemorhous": draw_finch,
    "Leucosticte": draw_finch,
    "Pinicola": draw_finch,
    "Eophona": draw_finch,
    "Mycerobas": draw_finch,
    "Melospiza": draw_finch,
    "Zonotrichia": draw_finch,
    "Junco": draw_finch,
    "Passerculus": draw_finch,
    "Ammodramus": draw_finch,
    "Calcarius": draw_finch,
    "Plectrophenax": draw_finch,
    "Miliaria": draw_finch,
    # Thrushes / robins / chats
    "Turdus": draw_thrush,
    "Erithacus": draw_thrush,
    "Phoenicurus": draw_thrush,
    "Luscinia": draw_thrush,
    "Oenanthe": draw_thrush,
    "Saxicola": draw_thrush,
    "Monticola": draw_thrush,
    "Catharus": draw_thrush,
    "Hylocichla": draw_thrush,
    "Sialia": draw_thrush,
    "Muscicapa": draw_thrush,
    "Ficedula": draw_thrush,
    "Copsychus": draw_thrush,
    "Kittacincla": draw_thrush,
    # Warblers
    "Phylloscopus": draw_warbler,
    "Sylvia": draw_warbler,
    "Regulus": draw_warbler,
    "Troglodytes": draw_warbler,
    "Curruca": draw_warbler,
    "Acrocephalus": draw_warbler,
    "Cettia": draw_warbler,
    "Hippolais": draw_warbler,
    "Locustella": draw_warbler,
    "Iduna": draw_warbler,
    "Setophaga": draw_warbler,
    "Dendroica": draw_warbler,
    "Vermivora": draw_warbler,
    "Geothlypis": draw_warbler,
    "Protonotaria": draw_warbler,
    "Leiothlypis": draw_warbler,
    "Cistus": draw_warbler,
    "Cisticola": draw_warbler,
    "Prinia": draw_warbler,
    "Calamanthus": draw_warbler,
    # Woodpeckers
    "Dendrocopos": draw_woodpecker,
    "Dryobates": draw_woodpecker,
    "Dryocopus": draw_woodpecker,
    "Picus": draw_woodpecker,
    "Jynx": draw_woodpecker,
    "Picoides": draw_woodpecker,
    "Colaptes": draw_woodpecker,
    "Melanerpes": draw_woodpecker,
    "Campephilus": draw_woodpecker,
    "Celeus": draw_woodpecker,
    "Sphyrapicus": draw_woodpecker,
    # Corvids
    "Corvus": draw_corvid,
    "Pica": draw_corvid,
    "Garrulus": draw_corvid,
    "Nucifraga": draw_corvid,
    "Cyanopica": draw_corvid,
    "Perisoreus": draw_corvid,
    "Aphelocoma": draw_corvid,
    "Cyanocitta": draw_corvid,
    "Gymnorhinus": draw_corvid,
    "Pyrrhocorax": draw_corvid,
    # Pigeons / doves
    "Columba": draw_pigeon,
    "Streptopelia": draw_pigeon,
    "Patagioenas": draw_pigeon,
    "Zenaida": draw_pigeon,
    "Columbina": draw_pigeon,
    "Geopelia": draw_pigeon,
    "Treron": draw_pigeon,
    # Swifts / swallows / martins
    "Hirundo": draw_swift_swallow,
    "Delichon": draw_swift_swallow,
    "Apus": draw_swift_swallow,
    "Riparia": draw_swift_swallow,
    "Ptyonoprogne": draw_swift_swallow,
    "Cecropis": draw_swift_swallow,
    "Tachycineta": draw_swift_swallow,
    "Progne": draw_swift_swallow,
    "Petrochelidon": draw_swift_swallow,
    # Nuthatches / treecreepers / wallcreeper
    "Sitta": draw_nuthatch_creeper,
    "Certhia": draw_nuthatch_creeper,
    "Tichodroma": draw_nuthatch_creeper,
    "Mniotilta": draw_nuthatch_creeper,
    # Wagtails / starlings / waxwings / pipits
    "Motacilla": draw_wagtail_starling,
    "Sturnus": draw_wagtail_starling,
    "Bombycilla": draw_wagtail_starling,
    "Anthus": draw_wagtail_starling,
    "Pastor": draw_wagtail_starling,
    "Acridotheres": draw_wagtail_starling,
    "Cinclus": draw_wagtail_starling,
    # Cuckoos
    "Cuculus": draw_cuckoo,
    "Clamator": draw_cuckoo,
    "Coccyzus": draw_cuckoo,
    "Crotophaga": draw_cuckoo,
}

# ---------------------------------------------------------------------------
# Colour conversion helpers (sRGB -> Lab)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    """Linearise one sRGB channel (0..1)."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Linear sRGB -> CIE XYZ (D65)."""
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z


def _f_lab(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return t / (3.0 * delta ** 2) + 4.0 / 29.0


def _rgb_to_lab(r_u8: int, g_u8: int, b_u8: int) -> tuple[float, float, float]:
    """uint8 RGB -> CIE L*a*b* (D65, no external deps)."""
    r = _srgb_to_linear(r_u8 / 255.0)
    g = _srgb_to_linear(g_u8 / 255.0)
    b = _srgb_to_linear(b_u8 / 255.0)
    x, y, z = _linear_to_xyz(r, g, b)
    # D65 white point
    fx = _f_lab(x / 0.95047)
    fy = _f_lab(y / 1.00000)
    fz = _f_lab(z / 1.08883)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return L, a, b_


def _pixels_to_lab_array(
    pixels: list[tuple[int, int, int]],
) -> list[tuple[float, float, float]]:
    """Convert a list of uint8 RGB tuples to Lab."""
    if _HAS_SKIMAGE and _HAS_NUMPY:
        from skimage.color import rgb2lab as sk_rgb2lab  # type: ignore[import]
        arr = np.array(pixels, dtype=np.float32) / 255.0
        arr = arr.reshape(1, -1, 3)
        lab = sk_rgb2lab(arr).reshape(-1, 3)
        return [tuple(row) for row in lab.tolist()]  # type: ignore[return-value]
    return [_rgb_to_lab(r, g, b) for r, g, b in pixels]


# ---------------------------------------------------------------------------
# K-means implementations
# ---------------------------------------------------------------------------

def _kmeans_sklearn(
    pixels: list[tuple[int, int, int]], k: int, n_init: int = 4
) -> list[tuple[tuple[int, int, int], float]]:
    """scikit-learn KMeans on Lab pixels. Returns [(rgb, fraction), ...]."""
    from sklearn.cluster import KMeans  # type: ignore[import]
    lab_pts = _pixels_to_lab_array(pixels)
    arr = np.array(lab_pts, dtype=np.float32)
    actual_k = min(k, len(arr))
    km = KMeans(n_clusters=actual_k, n_init=n_init, random_state=42)
    km.fit(arr)
    labels = km.labels_
    total = len(labels)
    result: list[tuple[tuple[int, int, int], float]] = []
    for ci in range(actual_k):
        mask = labels == ci
        fraction = float(mask.sum()) / total
        cluster_rgb = [pixels[i] for i in range(total) if mask[i]]
        if not cluster_rgb:
            continue
        mean_r = int(sum(p[0] for p in cluster_rgb) / len(cluster_rgb))
        mean_g = int(sum(p[1] for p in cluster_rgb) / len(cluster_rgb))
        mean_b = int(sum(p[2] for p in cluster_rgb) / len(cluster_rgb))
        result.append(((mean_r, mean_g, mean_b), fraction))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _kmeans_pure(
    pixels: list[tuple[int, int, int]], k: int, iterations: int = 20
) -> list[tuple[tuple[int, int, int], float]]:
    """
    Pure-Python k-means on quantised Lab pixels.
    Quantises each Lab channel to 32 bins first for speed.
    """
    def quantise(lab: tuple[float, float, float]) -> tuple[int, int, int]:
        L_q = int(max(0, min(31, lab[0] / 100.0 * 31)))
        a_q = int(max(0, min(31, (lab[1] + 128) / 256.0 * 31)))
        b_q = int(max(0, min(31, (lab[2] + 128) / 256.0 * 31)))
        return (L_q, a_q, b_q)

    lab_pixels = _pixels_to_lab_array(pixels)
    quant = [quantise(p) for p in lab_pixels]
    total = len(quant)

    # Seed centres using k-means++ style initialisation
    rng = random.Random(42)
    centres: list[tuple[int, int, int]] = [rng.choice(quant)]
    while len(centres) < min(k, total):
        dists = []
        for pt in quant:
            d = min(
                sum((pt[i] - c[i]) ** 2 for i in range(3))
                for c in centres
            )
            dists.append(d)
        total_d = sum(dists)
        if total_d == 0:
            break
        r = rng.random() * total_d
        cumulative = 0.0
        for pt, d in zip(quant, dists):
            cumulative += d
            if cumulative >= r:
                centres.append(pt)
                break

    for _ in range(iterations):
        clusters: dict[int, list[int]] = {i: [] for i in range(len(centres))}
        for idx, pt in enumerate(quant):
            best = min(
                range(len(centres)),
                key=lambda ci: sum((pt[j] - centres[ci][j]) ** 2 for j in range(3)),
            )
            clusters[best].append(idx)
        new_centres = []
        for ci, idxs in clusters.items():
            if not idxs:
                new_centres.append(centres[ci])
                continue
            mean = tuple(
                int(sum(quant[i][d] for i in idxs) / len(idxs))
                for d in range(3)
            )
            new_centres.append(mean)  # type: ignore[arg-type]
        if new_centres == centres:
            break
        centres = new_centres  # type: ignore[assignment]

    # Build result in original RGB
    result: list[tuple[tuple[int, int, int], float]] = []
    clusters_final: dict[int, list[int]] = {i: [] for i in range(len(centres))}
    for idx, pt in enumerate(quant):
        best = min(
            range(len(centres)),
            key=lambda ci: sum((pt[j] - centres[ci][j]) ** 2 for j in range(3)),
        )
        clusters_final[best].append(idx)

    for ci, idxs in clusters_final.items():
        if not idxs:
            continue
        fraction = len(idxs) / total
        cluster_rgb = [pixels[i] for i in idxs]
        mean_r = int(sum(p[0] for p in cluster_rgb) / len(cluster_rgb))
        mean_g = int(sum(p[1] for p in cluster_rgb) / len(cluster_rgb))
        mean_b = int(sum(p[2] for p in cluster_rgb) / len(cluster_rgb))
        result.append(((mean_r, mean_g, mean_b), fraction))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def dominant_colors(
    pixels: list[tuple[int, int, int]], k: int = 6
) -> list[tuple[tuple[int, int, int], float]]:
    """Return up to k (rgb_tuple, fraction) pairs, sorted by fraction desc."""
    if not pixels:
        return []
    if _HAS_SKLEARN and _HAS_NUMPY:
        return _kmeans_sklearn(pixels, k)
    return _kmeans_pure(pixels, k)


# ---------------------------------------------------------------------------
# Colour attribute helpers
# ---------------------------------------------------------------------------

def _lab_of(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return _rgb_to_lab(rgb[0], rgb[1], rgb[2])


def _lightness(rgb: tuple[int, int, int]) -> float:
    return _lab_of(rgb)[0]


def _chroma(rgb: tuple[int, int, int]) -> float:
    _, a, b = _lab_of(rgb)
    return math.sqrt(a * a + b * b)


def _hue_deg(rgb: tuple[int, int, int]) -> float:
    """Hue angle in degrees (0..360) in Lab space."""
    _, a, b = _lab_of(rgb)
    return math.degrees(math.atan2(b, a)) % 360.0


def _rgb_to_cairo(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0


def _darken(rgb: tuple[int, int, int], factor: float = 0.88) -> tuple[float, float, float]:
    return (
        max(0.0, rgb[0] / 255.0 * factor),
        max(0.0, rgb[1] / 255.0 * factor),
        max(0.0, rgb[2] / 255.0 * factor),
    )


def _is_warm(rgb: tuple[int, int, int]) -> bool:
    """True for yellow / orange / red hues."""
    h = _hue_deg(rgb)
    return h < 60.0 or h > 320.0


def _is_white_ish(rgb: tuple[int, int, int]) -> bool:
    return _lightness(rgb) > 82.0 and _chroma(rgb) < 18.0


def _is_dark(rgb: tuple[int, int, int]) -> bool:
    return _lightness(rgb) < 38.0


def _fallback_palette() -> list[tuple[tuple[int, int, int], float]]:
    """Generic grey-brown palette used when a class has no usable samples."""
    return [
        ((110, 85, 60), 0.30),
        ((80, 60, 40), 0.22),
        ((165, 145, 120), 0.18),
        ((220, 210, 195), 0.12),
        ((55, 45, 30), 0.10),
        ((190, 110, 50), 0.08),
    ]


# ---------------------------------------------------------------------------
# Step 3 — Color -> slot mapping
# ---------------------------------------------------------------------------

def colors_to_slots(
    palette: list[tuple[tuple[int, int, int], float]],
) -> dict[str, tuple[float, ...]]:
    """
    Map a sorted (rgb, fraction) palette onto template colour slots.

    Always returns at minimum: body, wing, belly, head, beak, feet.
    Additional slots (cap, cheek, stripe, breast) are added when the
    palette data supports them.
    """
    if not palette:
        palette = _fallback_palette()

    pal = sorted(palette, key=lambda x: x[1], reverse=True)
    rgbs = [p[0] for p in pal]

    # --- body: largest colour that is not extremely dark and not white-ish ---
    body_rgb = rgbs[0]
    for rgb, _ in pal:
        if not _is_white_ish(rgb) and not _is_dark(rgb):
            body_rgb = rgb
            break

    body = _rgb_to_cairo(body_rgb)
    wing = _darken(body_rgb, 0.88)

    # --- head: darkest among the top-4 colours ---
    top4 = rgbs[:min(4, len(rgbs))]
    head_rgb = min(top4, key=lambda r: _lightness(r))
    if head_rgb == body_rgb and len(rgbs) > 1:
        head_rgb = min(rgbs, key=lambda r: _lightness(r))
    head = _rgb_to_cairo(head_rgb)

    colors: dict[str, tuple[float, ...]] = {}

    # --- cap: same as head when head is quite dark ---
    if _lightness(head_rgb) < 50.0:
        colors["cap"] = head

    # --- belly: warm or light colour with decent fraction ---
    belly_rgb = body_rgb
    for rgb, frac in pal:
        if frac < 0.03:
            continue
        if _is_warm(rgb) and _chroma(rgb) > 22.0:
            belly_rgb = rgb
            break
        if _lightness(rgb) > 70.0 and not _is_white_ish(rgb):
            belly_rgb = rgb
            break
    belly = _rgb_to_cairo(belly_rgb)

    # --- breast: orange-red stripe if distinct from belly ---
    for rgb, frac in pal:
        if frac < 0.04:
            continue
        h = _hue_deg(rgb)
        if 5.0 <= h <= 50.0 and _chroma(rgb) > 28.0 and rgb != belly_rgb:
            colors["breast"] = _rgb_to_cairo(rgb)
            break

    # --- cheek / bar: white-ish small patch ---
    for rgb, frac in pal:
        if _is_white_ish(rgb) and frac < 0.25:
            colors["cheek"] = _rgb_to_cairo(rgb)
            break

    # --- stripe: narrow dark patch (e.g. great tit ventral stripe) ---
    for rgb, frac in pal:
        if _is_dark(rgb) and 0.04 < frac < 0.18 and rgb != head_rgb:
            colors["stripe"] = _rgb_to_cairo(rgb)
            break

    # --- beak: dark unless yellow/orange evidence found ---
    beak_rgb: tuple[int, int, int] = (55, 50, 45)
    for rgb, frac in pal:
        h = _hue_deg(rgb)
        if 35.0 < h < 75.0 and _chroma(rgb) > 30.0 and frac < 0.12:
            beak_rgb = rgb
            break
    beak = _rgb_to_cairo(beak_rgb)

    # --- feet: dark grey-brown (rarely visible in dataset crops) ---
    feet: tuple[float, ...] = (0.28, 0.22, 0.17)

    colors.update({
        "body": body,
        "wing": wing,
        "belly": belly,
        "head": head,
        "beak": beak,
        "feet": feet,
    })
    return colors


# ---------------------------------------------------------------------------
# Step 1 — Dataset parsing
# ---------------------------------------------------------------------------

def _find_images_and_labels(
    dataset_dir: Path,
) -> tuple[list[Path], Path | None]:
    """
    Return (image_paths, labels_root).
    labels_root is None for Layout B (labels alongside images).
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if images_dir.is_dir() and labels_dir.is_dir():
        # Layout A: walk all split subdirs
        img_paths: list[Path] = []
        for subdir in sorted(images_dir.iterdir()):
            if subdir.is_dir():
                img_paths.extend(subdir.glob("*.jpg"))
                img_paths.extend(subdir.glob("*.jpeg"))
                img_paths.extend(subdir.glob("*.png"))
        return img_paths, labels_dir

    # Layout B: flat
    img_paths = (
        list(dataset_dir.glob("*.jpg"))
        + list(dataset_dir.glob("*.jpeg"))
        + list(dataset_dir.glob("*.png"))
    )
    return img_paths, None


def _label_path_for_image(
    image_path: Path,
    labels_root: Path | None,
) -> Path:
    """Derive the .txt label path from an image path."""
    if labels_root is None:
        return image_path.with_suffix(".txt")
    # Layout A: mirror the split subfolder under labels_root
    try:
        return labels_root / image_path.parent.name / (image_path.stem + ".txt")
    except (ValueError, AttributeError):
        return labels_root / (image_path.stem + ".txt")


def _load_class_names(dataset_dir: Path) -> dict[int, str]:
    """Load class index -> name mapping from data.yaml or classes.txt."""
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists() and _HAS_YAML:
        import yaml as _yaml_mod  # type: ignore[import]
        with yaml_path.open() as fh:
            data = _yaml_mod.safe_load(fh)
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

    for txt_name in ("classes.txt", "obj.names"):
        txt_path = dataset_dir / txt_name
        if txt_path.exists():
            lines = txt_path.read_text().splitlines()
            return {i: ln.strip() for i, ln in enumerate(lines) if ln.strip()}

    for sub in ("images", ""):
        p = dataset_dir / sub / "classes.txt" if sub else dataset_dir / "classes.txt"
        if p.exists():
            lines = p.read_text().splitlines()
            return {i: ln.strip() for i, ln in enumerate(lines) if ln.strip()}

    raise FileNotFoundError(
        f"No class names found in {dataset_dir}. "
        "Expected data.yaml or classes.txt."
    )


def build_class_sample_map(
    dataset_dir: Path,
) -> tuple[
    dict[int, str],
    dict[int, list[tuple[Path, tuple[float, float, float, float]]]],
]:
    """
    Returns:
        class_names: {class_id: name}
        sample_map:  {class_id: [(image_path, (cx, cy, w, h)), ...]}
    """
    class_names = _load_class_names(dataset_dir)
    img_paths, labels_root = _find_images_and_labels(dataset_dir)

    sample_map: dict[int, list[tuple[Path, tuple[float, float, float, float]]]] = {
        cid: [] for cid in class_names
    }
    warned_missing: set[Path] = set()

    for img_path in img_paths:
        lbl_path = _label_path_for_image(img_path, labels_root)
        if not lbl_path.exists():
            if img_path not in warned_missing:
                log.debug("No label for %s", img_path)
                warned_missing.add(img_path)
            continue
        try:
            lines = lbl_path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            if cid not in sample_map:
                sample_map[cid] = []
            sample_map[cid].append((img_path, (cx, cy, bw, bh)))

    return class_names, sample_map


# ---------------------------------------------------------------------------
# Step 2 — Extract dominant colors per class
# ---------------------------------------------------------------------------

def _crop_pixels(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    max_side: int = 128,
) -> list[tuple[int, int, int]] | None:
    """Open image, crop to bbox, return list of (r,g,b) uint8 tuples."""
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

    iw, ih = img.size
    cx, cy, bw, bh = bbox
    x0 = int((cx - bw / 2) * iw)
    y0 = int((cy - bh / 2) * ih)
    x1 = int((cx + bw / 2) * iw)
    y1 = int((cy + bh / 2) * ih)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(iw, x1)
    y1 = min(ih, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    scale = max_side / max(cw, ch)
    if scale < 1.0:
        crop = crop.resize(
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            Image.Resampling.LANCZOS,
        )
    return list(crop.getdata())  # type: ignore[return-value]


def extract_colors_for_class(
    samples: list[tuple[Path, tuple[float, float, float, float]]],
    n_samples: int = 40,
    k: int = 6,
) -> list[tuple[tuple[int, int, int], float]]:
    """
    Sample up to n_samples bboxes, collect pixels, cluster into k colors.
    Returns [(rgb, fraction), ...] sorted by fraction desc.
    """
    chosen = samples if len(samples) <= n_samples else random.sample(samples, n_samples)
    all_pixels: list[tuple[int, int, int]] = []
    warned_paths: set[Path] = set()

    for img_path, bbox in chosen:
        if not img_path.exists():
            if img_path not in warned_paths:
                log.warning("Missing image: %s (skipping)", img_path)
                warned_paths.add(img_path)
            continue
        pixels = _crop_pixels(img_path, bbox)
        if pixels is None:
            log.warning("Corrupt or unreadable image: %s (skipping)", img_path)
            continue
        all_pixels.extend(pixels)

    if not all_pixels:
        log.warning("No usable pixels found — using fallback grey-brown palette")
        return _fallback_palette()

    # Subsample to keep clustering fast (max 20k pixels)
    if len(all_pixels) > 20_000:
        all_pixels = random.sample(all_pixels, 20_000)

    return dominant_colors(all_pixels, k=k)


# ---------------------------------------------------------------------------
# Step 4 — Template selection by genus
# ---------------------------------------------------------------------------

def _genus_from_name(class_name: str) -> str:
    """Extract genus from a scientific name like 'Parus_major' -> 'Parus'."""
    parts = class_name.split("_")
    if len(parts) >= 2 and parts[0] and parts[0][0].isupper():
        return parts[0]
    return class_name


def _safe_filename(class_name: str) -> str:
    """Return a filesystem-safe version of the class name."""
    return re.sub(r"[^\w\-.]", "_", class_name)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _sample_hash(
    samples: list[tuple[Path, tuple[float, float, float, float]]],
) -> str:
    """Stable hash of the (sorted) image paths used for a class."""
    paths = sorted(str(s[0]) for s in samples)
    return hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache_path: Path, data: dict[str, Any]) -> None:
    try:
        cache_path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        log.warning("Could not write cache: %s", exc)


# ---------------------------------------------------------------------------
# Per-class worker (called from ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_class_worker(
    args: tuple[
        str,                                                    # class_name
        list[tuple[str, tuple[float, float, float, float]]],   # samples (paths as str)
        int,                                                    # n_samples
        int,                                                    # k_clusters
        str,                                                    # out_dir as str
        str | None,                                             # cached_palette JSON or None
    ],
) -> tuple[str, list[list[Any]] | None, str]:
    """
    Worker executed per class.
    Returns (class_name, fresh_palette_or_None, status_message).
    fresh_palette is only set on a cache miss.
    """
    class_name, raw_samples, n_samples, k, out_dir_str, cached_palette_json = args

    samples = [(Path(p), bbox) for p, bbox in raw_samples]

    if cached_palette_json is not None:
        palette_data = json.loads(cached_palette_json)
        palette: list[tuple[tuple[int, int, int], float]] = [
            (tuple(item[0]), item[1])  # type: ignore[misc]
            for item in palette_data
        ]
        fresh: list[list[Any]] | None = None
    else:
        palette = extract_colors_for_class(samples, n_samples=n_samples, k=k)
        fresh = [[list(rgb), frac] for rgb, frac in palette]

    colors = colors_to_slots(palette)
    genus = _genus_from_name(class_name)
    template_fn = GENUS_TEMPLATES.get(genus, draw_finch)
    if genus not in GENUS_TEMPLATES:
        log.info("Unknown genus '%s' for %s — using draw_finch", genus, class_name)
    template_name = template_fn.__name__

    out_path = Path(out_dir_str) / (_safe_filename(class_name) + ".png")
    surface, ctx = new_surface()
    template_fn(ctx, colors)
    surface.write_to_png(str(out_path))

    msg = f"{len(palette)} colors, {template_name} -> {out_path.name}"
    return class_name, fresh, msg


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    dataset_dir: Path,
    out_dir: Path,
    n_samples: int = 40,
    k_clusters: int = 6,
    force: bool = False,
    only: set[str] | None = None,
    workers: int | None = None,
) -> None:
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "_color_profiles.json"

    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {out_dir}")
    if not _HAS_SKLEARN:
        print("INFO: scikit-learn not found — using pure-Python k-means (slower)")
    if not _HAS_SKIMAGE:
        print("INFO: scikit-image not found — using manual sRGB->Lab conversion")

    class_names, sample_map = build_class_sample_map(dataset_dir)
    cache: dict[str, Any] = {} if force else _load_cache(cache_path)

    classes_to_process = sorted(
        (cid, name)
        for cid, name in class_names.items()
        if only is None or name in only
    )
    total = len(classes_to_process)
    if total == 0:
        print("No classes to process.")
        return

    # Build worker args, serialising Paths to str for subprocess transport
    worker_args: list[tuple] = []
    for cid, class_name in classes_to_process:
        samples = sample_map.get(cid, [])
        cache_key = f"{class_name}:{_sample_hash(samples)}"
        cached_json: str | None = None
        if cache_key in cache and not force:
            cached_json = json.dumps(cache[cache_key])
        raw_samples = [(str(p), bbox) for p, bbox in samples]
        worker_args.append((class_name, raw_samples, n_samples, k_clusters, str(out_dir), cached_json))

    actual_workers = workers if workers is not None else (os.cpu_count() or 1)

    results: list[tuple[str, list[list[Any]] | None, str]] = []

    if actual_workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            future_to_meta = {
                pool.submit(_process_class_worker, arg): (i, arg[0])
                for i, arg in enumerate(worker_args)
            }
            done_count = 0
            for future in as_completed(future_to_meta):
                done_count += 1
                _, class_name = future_to_meta[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[{done_count:3d}/{total}] {class_name}  ERROR: {exc}")
                    continue
                results.append(result)
                _, _, msg = result
                print(f"[{done_count:3d}/{total}] {class_name}  {msg}")
    else:
        for idx, arg in enumerate(worker_args, start=1):
            class_name = arg[0]
            try:
                result = _process_class_worker(arg)
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx:3d}/{total}] {class_name}  ERROR: {exc}")
                continue
            results.append(result)
            _, _, msg = result
            print(f"[{idx:3d}/{total}] {class_name}  {msg}")

    # Persist newly computed palettes to cache
    for class_name, fresh_palette, _ in results:
        if fresh_palette is not None:
            matching_cid = next(
                (cid for cid, n in class_names.items() if n == class_name), None
            )
            if matching_cid is not None:
                samples = sample_map.get(matching_cid, [])
                cache_key = f"{class_name}:{_sample_hash(samples)}"
                cache[cache_key] = fresh_palette

    _save_cache(cache_path, cache)

    elapsed = time.monotonic() - t0
    print(f"\nDone. {total} icons in {elapsed:.1f}s  ->  {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_icons_from_dataset.py",
        description=(
            "Generate bold flat-art bird icons for every species "
            "in a YOLO training dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root folder of the YOLO dataset (Layout A or B).",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory for PNGs. Default: DATASET_DIR/_icons/",
    )
    parser.add_argument(
        "--samples",
        dest="n_samples",
        type=int,
        default=40,
        metavar="N",
        help="Max bbox samples per class for colour extraction (default: 40).",
    )
    parser.add_argument(
        "--clusters",
        dest="k_clusters",
        type=int,
        default=6,
        metavar="K",
        help="Number of dominant colours to extract per class (default: 6).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild colour cache from scratch.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="CLASS1,CLASS2",
        help="Comma-separated class names to process (others skipped).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel worker processes (default: os.cpu_count()).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        parser.error(f"Dataset directory not found: {dataset_dir}")

    out_dir: Path = (
        args.out_dir.resolve() if args.out_dir else dataset_dir / "_icons"
    )

    only: set[str] | None = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    run(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        n_samples=args.n_samples,
        k_clusters=args.k_clusters,
        force=args.force,
        only=only,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
