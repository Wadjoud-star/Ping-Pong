"""Point d'entrée : jeu ping-pong ± détection gestes webcam.

Exemples :
  # Clavier seul
  python -m src.main

  # Webcam + YOLO (fluide : 1 détection / 3 frames, MPS si dispo)
  python -m src.main --gestures --model models/best.pt --conf 0.25 --imgsz 320 --device mps

  # Avec aperçu YOLO (un peu plus lourd)
  python -m src.main --gestures --model models/best.pt --preview --detect-every 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permet `python src/main.py` depuis la racine du projet
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ping-Pong contrôlé par gestes (Lab 5)")
    p.add_argument("--gestures", action="store_true", help="Active la webcam + YOLO")
    p.add_argument("--source", default="0", help="Index caméra ou chemin vidéo")
    p.add_argument("--model", default="models/best.pt", help="Poids YOLO (.pt)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=320, help="Taille d'inférence YOLO (plus petit = plus rapide)")
    p.add_argument("--device", default="cpu", help="cpu, mps (Mac Apple Silicon), ou 0 (CUDA)")
    p.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="Lancer YOLO 1 frame sur N (défaut: 3). Plus grand = plus fluide",
    )
    p.add_argument("--preview", action="store_true", help="Fenêtre OpenCV annotée (plus lent)")
    p.add_argument("--mirror", action="store_true", default=True, help="Miroir webcam (défaut)")
    p.add_argument("--no-mirror", action="store_false", dest="mirror")
    return p.parse_args()


def run_keyboard() -> int:
    from src.game.pong import PongGame

    PongGame().run_keyboard_only()
    return 0


def _controls_from_gesture(gf) -> "ControlState":
    from src.game.pong import ControlState

    return ControlState(
        p1_y=gf.p1_y,
        p2_y=gf.p2_y,
        p1_hit=gf.p1_hit,
        p2_hit=gf.p2_hit,
        labels={
            "p1": f"fist ({gf.p1_y:.2f})" if gf.p1_y is not None else "—",
            "p2": f"open_hand ({gf.p2_y:.2f})" if gf.p2_y is not None else "—",
        },
    )


def run_with_gestures(args: argparse.Namespace) -> int:
    import cv2

    from src.detection.gestures import GestureDetector
    from src.game.pong import ControlState, PongGame

    model_path = Path(args.model)
    if not model_path.is_file() and args.model != "yolo11n.pt":
        # Fallback pratique avant le fine-tune
        print(f"[warn] Modèle introuvable: {model_path}")
        print("[warn] Fallback sur yolo11n.pt (classes COCO — gestes custom absents).")
        print("[warn] Après entraînement, placez best.pt dans models/best.pt")
        args.model = "yolo11n.pt"

    if args.detect_every < 1:
        print("--detect-every must be >= 1")
        return 2

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam/vidéo. Essayez --source 1")
        return 1

    # Webcam un peu plus légère (suffisant pour les gestes)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = GestureDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    game = PongGame(title="Ping-Pong | Gestures + YOLO")

    frame_i = 0
    last_controls = ControlState()
    last_preview = None

    print("Contrôles: W/S et ↑/↓ toujours actifs. Q/ESC pour quitter.")
    print(f"Perf: YOLO 1/{args.detect_every} frame(s), imgsz={args.imgsz}, device={args.device}")
    try:
        while game.running:
            ok, frame = cap.read()
            if not ok:
                print("Fin du flux vidéo.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            # YOLO seulement 1 frame sur N ; entre-temps on réutilise le dernier geste
            if frame_i % args.detect_every == 0:
                gf = detector.predict(frame, annotate=args.preview)
                last_controls = _controls_from_gesture(gf)
                if args.preview:
                    last_preview = gf.annotated_bgr
            else:
                # Ne pas re-déclencher les "hits" sur les frames sautées
                last_controls = ControlState(
                    p1_y=last_controls.p1_y,
                    p2_y=last_controls.p2_y,
                    p1_hit=False,
                    p2_hit=False,
                    labels=last_controls.labels,
                )

            game.set_controls(last_controls)
            frame_i += 1

            if args.preview and last_preview is not None:
                try:
                    cv2.imshow("YOLO gestures (q in game to quit)", last_preview)
                    cv2.waitKey(1)
                except cv2.error as ex:
                    print("Fenêtre OpenCV indisponible:", ex)
                    args.preview = False

            if not game.tick():
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        import pygame

        pygame.quit()
    return 0


def main() -> int:
    args = parse_args()
    if not 0 <= args.conf <= 1:
        print("--conf must be between 0 and 1")
        return 2
    if args.gestures:
        return run_with_gestures(args)
    return run_keyboard()


if __name__ == "__main__":
    raise SystemExit(main())
