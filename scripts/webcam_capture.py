"""Capture d'images webcam pour annotation CVAT (format YOLO Detection).

SPACE = sauvegarder | Q / ESC = quitter
Sessions séparées recommandées : train / val / test
  python scripts/webcam_capture.py --output captures/train
  python scripts/webcam_capture.py --output captures/val
  python scripts/webcam_capture.py --output captures/test
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture webcam pour dataset gestes")
    parser.add_argument("--source", default="0", help="Index caméra ou fichier vidéo")
    parser.add_argument("--output", default="captures", help="Dossier de sortie")
    parser.add_argument("--mirror", action="store_true", help="Afficher/miroiter (sauve l'original)")
    args = parser.parse_args()

    output = Path(args.output)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Impossible d'ouvrir l'entrée. Essayez --source 1 ou un fichier vidéo.")
        return 1

    count = 0
    print("SPACE = sauver | Q/ESC = quitter")
    print("Variez angles, distances, éclairages, fonds. Consentement si visages visibles.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            display = cv2.flip(frame, 1) if args.mirror else frame.copy()
            cv2.putText(
                display,
                f"SPACE: save | Q: quit | saved={count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            try:
                cv2.imshow("Ping-Pong | Capture gestes", display)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as ex:
                print("Aperçu OpenCV impossible:", ex)
                return 2

            if key == ord(" "):
                output.mkdir(parents=True, exist_ok=True)
                # Sauvegarde sans overlay ; miroir optionnel pour cohérence démo
                to_save = cv2.flip(frame, 1) if args.mirror else frame
                path = output / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(path), to_save)
                count += 1
                print("Saved", path)
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"{count} image(s) sauvegardée(s) → {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
