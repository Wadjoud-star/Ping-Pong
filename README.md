# Ping-Pong contrôlé par gestes (ESIGELEC — Lab 5)

Jeu de ping-pong à l’écran. Les gestes détectés par la webcam (YOLO fine-tuné) pilotent les raquettes.

| Joueur | Geste YOLO | Action |
|--------|------------|--------|
| P1 (gauche) | `fist` | Position Y de la raquette (+ hit à l’apparition) |
| P2 (droite) | `open_hand` | Position Y de la raquette (+ hit à l’apparition) |

Pipeline : **Webcam → YOLO11n (classes custom) → actions Pygame**

## Structure

```text
Ping-Pong/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py              # Entrée jeu ± gestes
│   ├── game/pong.py         # Ping-pong minimal
│   └── detection/gestures.py
├── scripts/
│   ├── play.py
│   ├── webcam_capture.py    # Collecte images (CVAT)
│   ├── webcam_detection.py  # Démo détection live
│   └── train_custom_yolo.py # Fine-tune
├── data/
│   ├── data.yaml            # Classes fist / open_hand
│   └── dataset/images|labels/{train,val,test}/
├── captures/                # Photos brutes avant CVAT
└── models/                  # best.pt après entraînement
```

Aligné sur le lab cours `AIWithPythonESIGELEC/lab05` (`webcam_capture`, `webcam_detection`, `train_custom_yolo`).

## Setup

```bash
cd Ping-Pong
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Vous pouvez aussi réutiliser le `.venv` du cours et y installer `pygame`.

## Jouer tout de suite (clavier)

```bash
python -m src.main
# ou
python scripts/play.py
```

- **P1** : `W` / `S`
- **P2** : `↑` / `↓`
- `R` reset score · `Q` / `ESC` quitter

## Collecte & annotation (CVAT)

Sessions **séparées** train / val / test :

```bash
python scripts/webcam_capture.py --output captures/train --mirror
python scripts/webcam_capture.py --output captures/val --mirror
python scripts/webcam_capture.py --output captures/test --mirror
```

1. Importer les images dans **CVAT Online** (task YOLO Detection).
2. Annoter les classes **`fist`** et **`open_hand`** (bounding boxes).
3. Exporter au format **YOLO 1.1** / Ultralytics.
4. Placer images + labels dans `data/dataset/images/{split}` et `data/dataset/labels/{split}`.
5. Vérifier `data/data.yaml` (`names: 0: fist, 1: open_hand`).

Conseil : fonds / angles / éclairages variés ; éviter d’uploader des visages identifiables sans consentement.

## Fine-tune YOLO11n

```bash
python scripts/train_custom_yolo.py --data data/data.yaml --epochs 30 --copy-best
```

Puis démo détection :

```bash
python scripts/webcam_detection.py --model models/best.pt --conf 0.35
```

## Démo live : jeu + gestes

```bash
python -m src.main --gestures --model models/best.pt --preview
```

Sans `models/best.pt`, le script retombe sur `yolo11n.pt` (classes COCO — utile pour tester le branchement webcam, pas les gestes custom).

## Répartition équipe (3)

1. **Dataset** — capture, CVAT, splits train/val/test, `data.yaml`
2. **Modèle** — fine-tune, métriques, `models/best.pt`
3. **Jeu / démo** — Pygame, mapping gestes, présentation live

## Roadmap courte

- [x] Structure repo
- [x] Jeu ping-pong clavier
- [x] Module détection + branchement `--gestures`
- [ ] Collecte / annotation CVAT
- [ ] Fine-tune + `models/best.pt`
- [ ] Démo présentation
