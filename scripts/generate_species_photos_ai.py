#!/usr/bin/env python3
"""Generate AI bird photos with Stable Diffusion / FLUX from a species list.

Future-proof: species + descriptions are read from JSON files, not hardcoded.
Device-adaptive: CUDA on Linux/Win, MPS on Apple Silicon, CPU fallback.

Usage examples
--------------

# Render every species in common_names_DE.json with SD 1.5 on M1
python scripts/generate_species_photos_ai.py \\
    --species-file assets/common_names_DE.json --model sd15

# Single species
python scripts/generate_species_photos_ai.py \\
    --species Parus_major --model sd15

# RTX 3090 with FLUX schnell (Apache 2.0, kommerziell frei)
python scripts/generate_species_photos_ai.py \\
    --species-file assets/common_names_DE.json --model flux-schnell

# Override prompt template / seed / steps
python scripts/generate_species_photos_ai.py \\
    --species Parus_major --steps 30 --seed 7 --style sticker
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import parse, request

try:
    import torch
except ImportError:
    sys.exit("torch missing.  pip install torch diffusers transformers accelerate safetensors")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model registry — all entries below are commercially usable
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    hf_id: str
    pipeline_class: str
    license_short: str
    default_steps: int
    default_guidance: float
    resolution: int
    variant: Optional[str] = None  # "fp16" if HF has a fp16 branch

MODELS: dict[str, ModelSpec] = {
    "sd15": ModelSpec(
        name="Stable Diffusion 1.5",
        hf_id="runwayml/stable-diffusion-v1-5",
        pipeline_class="StableDiffusionPipeline",
        license_short="CreativeML Open RAIL-M",
        default_steps=25, default_guidance=7.5, resolution=512,
    ),
    "sdxl": ModelSpec(
        name="Stable Diffusion XL 1.0 base",
        hf_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_class="StableDiffusionXLPipeline",
        license_short="CreativeML Open RAIL++-M",
        default_steps=30, default_guidance=7.0, resolution=1024,
        variant="fp16",
    ),
    "flux-schnell": ModelSpec(
        name="FLUX.1 [schnell]",
        hf_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        license_short="Apache 2.0",
        default_steps=4, default_guidance=0.0, resolution=1024,
    ),
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STYLE_STICKER = ("flat sticker illustration, bold black outline, "
                 "saturated vibrant colors, Risograph print style, "
                 "pure white background, centered, cute, modern children's book, "
                 "no shadow, no scenery")

STYLE_PHOTO = ("professional wildlife photograph, soft natural light, sharp focus, "
               "shallow depth of field, pure white seamless studio background, "
               "centered, no shadow, high detail")

NEGATIVE_BASE = ("text, watermark, signature, logo, multiple birds, "
                 "low quality, blurry, ugly, distorted, deformed")
NEGATIVE_STICKER = NEGATIVE_BASE + ", photo, photograph, 3d render, scenery, background"
NEGATIVE_PHOTO   = NEGATIVE_BASE + ", cartoon, drawing, illustration, painting"

# ---------------------------------------------------------------------------
# iNaturalist taxon lookup with on-disk cache
# ---------------------------------------------------------------------------

INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
DEFAULT_CACHE = REPO_ROOT / "data" / "species_prompts_cache.json"


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def lookup_inat(scientific_name: str) -> dict:
    """Return {'common_en': str, 'genus': str, 'rank': str} for a scientific name."""
    q = parse.quote(scientific_name)
    url = f"{INAT_TAXA}?q={q}&rank=species"
    req = request.Request(url, headers={"User-Agent": "WatchMyBirds/1.0"})
    try:
        with request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except Exception:
        return {}
    results = data.get("results", [])
    for r in results:
        if r.get("name", "").lower() == scientific_name.lower():
            return {
                "common_en": r.get("preferred_common_name", "") or "",
                "genus": (r.get("name") or "").split(" ")[0],
                "rank": r.get("rank", "species"),
            }
    if results:
        r = results[0]
        return {
            "common_en": r.get("preferred_common_name", "") or "",
            "genus": (r.get("name") or "").split(" ")[0],
            "rank": r.get("rank", "species"),
        }
    return {}


def build_prompt_for(scientific_key: str, style: str,
                     cache: dict, common_de_name: str = "") -> str:
    """Return one prompt string for a given species key."""
    sci = scientific_key.replace("_", " ")
    info = cache.get(scientific_key)
    if not info:
        info = lookup_inat(sci)
        cache[scientific_key] = info
    common_en = info.get("common_en", "").strip()

    if common_en:
        subject = f"a {common_en} ({sci}), single bird, side view"
    elif common_de_name:
        subject = f"a {common_de_name} ({sci}), single bird, side view"
    else:
        subject = f"a {sci}, single bird, side view"

    style_part = STYLE_STICKER if style == "sticker" else STYLE_PHOTO
    return f"{subject}, {style_part}"


# ---------------------------------------------------------------------------
# Device + pipeline
# ---------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(spec: ModelSpec, device: str):
    import diffusers
    cls = getattr(diffusers, spec.pipeline_class)
    # MPS + fp16 produces black images on M1 due to Metal numerical bugs.
    # Use fp32 there; CUDA can use fp16, CPU stays fp32.
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    kwargs: dict = {"torch_dtype": dtype, "use_safetensors": True}
    if spec.variant and device == "cuda":
        kwargs["variant"] = spec.variant
    if spec.pipeline_class == "StableDiffusionPipeline":
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False
    pipe = cls.from_pretrained(spec.hf_id, **kwargs)
    pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


# ---------------------------------------------------------------------------
# Species list loader
# ---------------------------------------------------------------------------

def load_species(args) -> dict[str, str]:
    """Return {scientific_key: common_de_name}."""
    if args.species:
        return {k: "" for k in args.species}
    if args.species_file:
        raw = json.loads(args.species_file.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items()
                if k and k[0].isupper() and k != "Unknown_species"}
    sys.exit("Pass --species KEY [KEY...] or --species-file path/to/common_names_*.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODELS), default="sd15")
    p.add_argument("--style", choices=["sticker", "photo"], default="sticker")
    p.add_argument("--species", nargs="+", help="Scientific keys (Parus_major)")
    p.add_argument("--species-file", type=Path,
                   help="JSON file mapping scientific_key -> common_name")
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/ai_birds"))
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                   help="iNat lookup cache (JSON)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Render only the first N species (debugging)")
    args = p.parse_args(argv)

    spec = MODELS[args.model]
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)

    targets = load_species(args)
    if args.limit:
        targets = dict(list(targets.items())[: args.limit])

    cache = load_cache(args.cache)

    steps = args.steps or spec.default_steps
    guidance = args.guidance if args.guidance is not None else spec.default_guidance
    resolution = args.resolution or spec.resolution
    negative = NEGATIVE_STICKER if args.style == "sticker" else NEGATIVE_PHOTO

    print(f"Model:       {spec.name}  ({spec.license_short})")
    print(f"Device:      {device}    Resolution: {resolution}x{resolution}    Steps: {steps}")
    print(f"Style:       {args.style}")
    print(f"Output dir:  {args.out}")
    print(f"Species:     {len(targets)}")
    print()

    t0 = time.time()
    pipe = build_pipeline(spec, device)
    print(f"Pipeline loaded in {time.time()-t0:.1f}s")

    for i, (key, common_de) in enumerate(targets.items(), 1):
        prompt = build_prompt_for(key, args.style, cache, common_de)
        save_cache(args.cache, cache)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        t = time.time()
        kw: dict = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=resolution,
            height=resolution,
            generator=gen,
        )
        if args.model != "flux-schnell":
            kw["negative_prompt"] = negative
        img = pipe(**kw).images[0]
        path = args.out / f"{key}.png"
        img.save(path)
        print(f"  [{i:3d}/{len(targets)}] {key:32s} -> {path.name}  ({time.time()-t:.1f}s)")

    save_cache(args.cache, cache)
    print(f"\nDone in {time.time()-t0:.1f}s.   Cache: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
