"""
vision/train.py
Train YOLOv11s-seg on the EcoStream AI augmented waste dataset.

Trains from the Ultralytics pretrained yolo11s-seg.pt checkpoint using
transfer learning. Saves the best weights to vision/model/best.pt.

Usage:
    python vision/train.py
    python vision/train.py --epochs 100 --batch 8 --device 0

Requirements:
    dataset/augmented/data.yaml must exist (run dataset/gan_mix.py first)
    GPU with >= 4GB VRAM recommended (RTX 3050 or better)

Output:
    vision/model/best.pt   — best checkpoint by val mask mAP50-95
    vision/model/last.pt   — final epoch checkpoint
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_YAML   = Path("dataset/augmented/data.yaml")
MODEL_DIR   = Path("vision/model")
RUNS_DIR    = Path("vision/model/runs")
RUN_NAME    = "seg_small"


def train(epochs: int, batch: int, device: str, imgsz: int, resume: bool) -> None:
    if not DATA_YAML.exists():
        log.error("data.yaml not found at %s", DATA_YAML.resolve())
        log.error("Run dataset/gan_mix.py first to generate the dataset.")
        sys.exit(1)

    from ultralytics import YOLO

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = "yolo11s-seg.pt"
    last_pt    = RUNS_DIR / RUN_NAME / "weights" / "last.pt"

    if resume and last_pt.exists():
        log.info("Resuming from %s", last_pt)
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        log.info("Starting training from %s", checkpoint)
        model = YOLO(checkpoint)
        model.train(
            task="segment",
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=4,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.3,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            verbose=True,
        )

    # Copy best weights to vision/model/best.pt
    best_src = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    best_dst = MODEL_DIR / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        log.info("Best weights saved -> %s", best_dst)
    else:
        log.warning("best.pt not found at %s — check training output", best_src)

    # Copy last weights
    last_src = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
    last_dst = MODEL_DIR / "last.pt"
    if last_src.exists():
        shutil.copy2(last_src, last_dst)
        log.info("Last weights saved -> %s", last_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv11s-seg on EcoStream AI waste dataset."
    )
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--device", type=str,   default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if available")
    args = parser.parse_args()
    train(args.epochs, args.batch, args.device, args.imgsz, args.resume)
