"""Démo détection live YOLO (gestes ou modèle générique).

  python scripts/webcam_detection.py --model models/best.pt --conf 0.35
  python scripts/webcam_detection.py --model yolo11n.pt --source 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.gestures import GestureDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Détection gestes YOLO live")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", action="store_false", dest="mirror")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    model = args.model
    if not Path(model).is_file() and model != "yolo11n.pt":
        print(f"[warn] {model} introuvable → yolo11n.pt")
        model = "yolo11n.pt"

    source = int(args.source) if str(args.source).isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        parser.error(f"Vidéo introuvable: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Impossible d'ouvrir caméra/vidéo.")
        return 1

    detector = GestureDetector(model_path=model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    print("q / ESC pour quitter.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            gf = detector.predict(frame, annotate=True)
            view = gf.annotated_bgr if gf.annotated_bgr is not None else frame
            try:
                cv2.imshow("Ping-Pong | YOLO detection", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            except cv2.error as ex:
                print("Fenêtre OpenCV impossible:", ex)
                return 2
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
