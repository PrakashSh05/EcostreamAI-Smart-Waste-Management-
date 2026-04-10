"""
dataset/gan_mix.py
Synthetic waste dataset generator for EcoStream AI.

Composites TACO + TrashNet waste images onto background images
to produce a YOLO-format instance segmentation dataset with
deliberately overlapping objects (simulating cluttered Indian waste bins).

Output structure:
    dataset/augmented/
    ├── images/
    │   ├── train/   (70%)
    │   ├── val/     (15%)
    │   └── test/    (15%)
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

Usage:
    python dataset/gan_mix.py
    python dataset/gan_mix.py --num-images 5000 --seed 42

Classes (index matches data.yaml):
    0: plastic   1: metal   2: food_waste
    3: paper     4: glass   5: cardboard
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["plastic", "metal", "food_waste", "paper", "glass", "cardboard"]
CLASS_MAP   = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE    = 640       # output image size (square)
MIN_ITEMS   = 2         # min waste items per scene
MAX_ITEMS   = 6         # max waste items per scene (forces overlap)
MIN_SCALE   = 0.12      # item scale relative to image width
MAX_SCALE   = 0.35

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

OUT_DIR     = Path("dataset/augmented")
RAW_DIR     = Path("dataset/raw")
ANNO_DIR    = Path("dataset/annotations")


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------
def _load_sources() -> tuple[list[tuple[Path, str]], list[Path]]:
    """
    Returns:
        items  — list of (image_path, class_name)
        backgrounds — list of background image paths
    """
    items: list[tuple[Path, str]] = []

    # TACO annotations (COCO format)
    taco_ann = ANNO_DIR / "taco_annotations.json"
    if taco_ann.exists():
        with open(taco_ann, encoding="utf-8") as f:
            coco = json.load(f)
        id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        cat_to_name: dict[int, str] = {}
        for cat in coco.get("categories", []):
            name = cat["name"].lower().replace(" ", "_")
            for cls in CLASS_NAMES:
                if cls in name:
                    cat_to_name[cat["id"]] = cls
                    break
        for ann in coco.get("annotations", []):
            cls = cat_to_name.get(ann.get("category_id", -1))
            if cls:
                img_path = RAW_DIR / "taco" / id_to_file.get(ann["image_id"], "")
                if img_path.exists():
                    items.append((img_path, cls))
        log.info("TACO: loaded %d item references", len(items))

    # TrashNet directory structure: raw/trashnet/<class>/*.jpg
    trashnet_root = RAW_DIR / "trashnet"
    if trashnet_root.exists():
        for cls in CLASS_NAMES:
            cls_dir = trashnet_root / cls
            if cls_dir.exists():
                for p in cls_dir.glob("*.jpg"):
                    items.append((p, cls))
                for p in cls_dir.glob("*.png"):
                    items.append((p, cls))
        log.info("TrashNet: %d total items after merge", len(items))

    # Fallback: scan raw/ for any labelled sub-folders
    if not items:
        log.warning("No TACO/TrashNet data found — scanning raw/ for class folders")
        for cls in CLASS_NAMES:
            for ext in ("jpg", "jpeg", "png"):
                for p in RAW_DIR.rglob(f"*.{ext}"):
                    if cls in p.parts:
                        items.append((p, cls))

    # Backgrounds
    bg_dir = RAW_DIR / "backgrounds"
    backgrounds = []
    if bg_dir.exists():
        backgrounds = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.png"))
    if not backgrounds:
        log.warning("No backgrounds found in %s — using solid grey", bg_dir)

    return items, backgrounds


# ---------------------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------------------
def _random_background(backgrounds: list[Path], rng: random.Random) -> Image.Image:
    if backgrounds:
        bg = Image.open(rng.choice(backgrounds)).convert("RGB")
        bg = bg.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    else:
        colour = rng.randint(80, 200)
        bg = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (colour, colour, colour))
    return bg


def _paste_item(
    bg: Image.Image,
    item_path: Path,
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[float, float]]] | None:
    """
    Load and paste one waste item onto bg.
    Returns (updated_bg, polygon_normalised) or None on failure.
    Polygon is list of (x, y) pairs normalised to [0, 1].
    """
    try:
        item = Image.open(item_path).convert("RGBA")
    except Exception:
        return None

    scale  = rng.uniform(MIN_SCALE, MAX_SCALE)
    w      = max(10, int(IMG_SIZE * scale))
    aspect = item.height / max(item.width, 1)
    h      = max(10, int(w * aspect))
    item   = item.resize((w, h), Image.LANCZOS)

    # Random position (allow partial overlap at edges)
    x0 = rng.randint(-w // 4, IMG_SIZE - w // 4)
    y0 = rng.randint(-h // 4, IMG_SIZE - h // 4)

    # Slight random rotation
    angle = rng.uniform(-30, 30)
    item  = item.rotate(angle, expand=True, resample=Image.BICUBIC)
    w, h  = item.size

    # Paste with alpha
    bg.paste(item, (x0, y0), item if item.mode == "RGBA" else None)

    # Build bounding polygon (8-point box) in normalised coords
    pts = [
        (x0,      y0),
        (x0 + w,  y0),
        (x0 + w,  y0 + h),
        (x0,      y0 + h),
    ]
    norm = [
        (
            max(0.0, min(1.0, px / IMG_SIZE)),
            max(0.0, min(1.0, py / IMG_SIZE)),
        )
        for px, py in pts
    ]
    # Ensure polygon has enough area
    xs = [p[0] for p in norm]
    ys = [p[1] for p in norm]
    if (max(xs) - min(xs)) < 0.01 or (max(ys) - min(ys)) < 0.01:
        return None

    return bg, norm


def _poly_to_yolo(cls_id: int, polygon: list[tuple[float, float]]) -> str:
    """Convert class id + polygon to YOLO segmentation label line."""
    pts_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{cls_id} {pts_str}"


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------
def generate_scene(
    items: list[tuple[Path, str]],
    backgrounds: list[Path],
    rng: random.Random,
) -> tuple[Image.Image, list[str]] | None:
    """
    Generate one synthetic scene.
    Returns (PIL image, list of YOLO label lines) or None if insufficient items.
    """
    if not items:
        return None

    bg         = _random_background(backgrounds, rng)
    n          = rng.randint(MIN_ITEMS, MAX_ITEMS)
    selected   = rng.choices(items, k=n)
    label_lines: list[str] = []

    for item_path, cls_name in selected:
        result = _paste_item(bg, item_path, rng)
        if result is None:
            continue
        bg, polygon = result
        cls_id = CLASS_MAP.get(cls_name, 0)
        label_lines.append(_poly_to_yolo(cls_id, polygon))

    if not label_lines:
        return None

    # Light post-processing: slight blur to blend composites
    bg = bg.filter(ImageFilter.GaussianBlur(radius=0.4))
    return bg.convert("RGB"), label_lines


# ---------------------------------------------------------------------------
# Dataset split and write
# ---------------------------------------------------------------------------
def build_dataset(num_images: int, seed: int) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    items, backgrounds = _load_sources()
    if not items:
        log.error(
            "No source waste images found. "
            "Place images in dataset/raw/trashnet/<class>/ "
            "or provide TACO annotations in dataset/annotations/."
        )
        sys.exit(1)

    log.info(
        "Generating %d scenes from %d items and %d backgrounds",
        num_images, len(items), len(backgrounds),
    )

    # Prepare output directories
    for split in SPLIT_RATIOS:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Assign splits
    indices  = list(range(num_images))
    rng.shuffle(indices)
    n_train  = int(num_images * SPLIT_RATIOS["train"])
    n_val    = int(num_images * SPLIT_RATIOS["val"])
    splits   = (
        ["train"] * n_train
        + ["val"]   * n_val
        + ["test"]  * (num_images - n_train - n_val)
    )

    generated = 0
    attempted = 0
    counters  = {"train": 0, "val": 0, "test": 0}

    while generated < num_images:
        attempted += 1
        if attempted > num_images * 5:
            log.warning("Too many failures — stopping at %d images", generated)
            break

        result = generate_scene(items, backgrounds, rng)
        if result is None:
            continue

        img, labels = result
        split       = splits[generated]
        idx         = counters[split]
        stem        = f"{split}_{idx:06d}"

        img_path = OUT_DIR / "images" / split / f"{stem}.jpg"
        lbl_path = OUT_DIR / "labels" / split / f"{stem}.txt"

        img.save(img_path, "JPEG", quality=90)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")

        counters[split] += 1
        generated       += 1

        if generated % 500 == 0:
            log.info("  %d / %d images generated", generated, num_images)

    # Write data.yaml
    data_yaml = OUT_DIR / "data.yaml"
    yaml_content = (
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    total = sum(counters.values())
    log.info(
        "Dataset complete: %d images  train=%d  val=%d  test=%d",
        total, counters["train"], counters["val"], counters["test"],
    )
    log.info("data.yaml written to %s", data_yaml)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic waste segmentation dataset."
    )
    parser.add_argument(
        "--num-images", type=int, default=3000,
        help="Total images to generate (default: 3000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    build_dataset(args.num_images, args.seed)