"""Fine-tune YOLO11n sur le dataset gestes (fist / open_hand).

Prérequis : images annotées CVAT exportées en YOLO Detection sous data/dataset/
avec data/data.yaml à jour.

Exemple (CPU, lab) :
  python scripts/train_custom_yolo.py --data data/data.yaml --epochs 30

Les poids finaux seront typiquement dans runs/detect/train/weights/best.pt
→ copiez-les vers models/best.pt pour la démo.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO gestes Ping-Pong")
    parser.add_argument("--data", default="data/data.yaml", help="Chemin data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolo11n.pt", help="Checkpoint de départ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--copy-best",
        action="store_true",
        help="Copie best.pt vers models/best.pt après entraînement",
    )
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_file():
        parser.error(f"data.yaml introuvable: {data}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="gestures",
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Entraînement terminé. best → {best}")
    if args.copy_best and best.is_file():
        dest = Path("models/best.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"Copié vers {dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
