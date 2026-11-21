"""Détection de gestes mains via Ultralytics YOLO.

Classes attendues (data/data.yaml) :
  0: fist       → Joueur 1
  1: open_hand  → Joueur 2

Sans poids custom, le détecteur peut tourner en mode "mock" (aucune détection)
ou avec un modèle générique (utile pour tester le pipeline webcam).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CLASS_FIST = "fist"
CLASS_OPEN_HAND = "open_hand"

# Mapping fixe aligné sur data/data.yaml
NAME_TO_PLAYER = {
    CLASS_FIST: 1,
    CLASS_OPEN_HAND: 2,
}


@dataclass
class Detection:
    name: str
    conf: float
    # xyxy pixels
    x1: float
    y1: float
    x2: float
    y2: float
    norm_y: float = 0.5  # centre Y ∈ [0, 1] (hauteur image)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class GestureFrame:
    """Résultat d'une frame pour brancher le jeu."""

    p1_y: float | None = None  # fist
    p2_y: float | None = None  # open_hand
    p1_hit: bool = False
    p2_hit: bool = False
    detections: list[Detection] = field(default_factory=list)
    annotated_bgr: np.ndarray | None = None


class GestureDetector:
    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        conf: float = 0.35,
        imgsz: int = 416,
        device: str = "cpu",
        class_names: dict[int, str] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # Noms issus du modèle fine-tuné, sinon mapping projet
        self.class_names = class_names or {
            0: CLASS_FIST,
            1: CLASS_OPEN_HAND,
        }
        self._prev_seen = {CLASS_FIST: False, CLASS_OPEN_HAND: False}

    def predict(self, frame_bgr: np.ndarray, annotate: bool = True) -> GestureFrame:
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        names = result.names or self.class_names

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                name = names.get(cls_id, self.class_names.get(cls_id, str(cls_id)))
                if name not in NAME_TO_PLAYER:
                    continue
                x1, y1, x2, y2 = map(float, box)
                cy = (y1 + y2) / 2
                detections.append(
                    Detection(
                        name=name,
                        conf=float(conf),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        norm_y=cy / max(h, 1),
                    )
                )

        # Meilleure conf par classe
        best: dict[str, Detection] = {}
        for det in detections:
            if det.name not in best or det.conf > best[det.name].conf:
                best[det.name] = det

        fist = best.get(CLASS_FIST)
        open_hand = best.get(CLASS_OPEN_HAND)

        # Hit = apparition (front montant) de la classe
        p1_hit = fist is not None and not self._prev_seen[CLASS_FIST]
        p2_hit = open_hand is not None and not self._prev_seen[CLASS_OPEN_HAND]
        self._prev_seen[CLASS_FIST] = fist is not None
        self._prev_seen[CLASS_OPEN_HAND] = open_hand is not None

        annotated = None
        if annotate:
            annotated = result.plot()
            # Overlay légende projet
            cv2.putText(
                annotated,
                "fist=P1 | open_hand=P2",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return GestureFrame(
            p1_y=fist.norm_y if fist else None,
            p2_y=open_hand.norm_y if open_hand else None,
            p1_hit=p1_hit,
            p2_hit=p2_hit,
            detections=detections,
            annotated_bgr=annotated,
        )
