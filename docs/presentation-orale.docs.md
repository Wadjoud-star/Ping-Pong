# Ping-Pong contrôlé par gestes — Script de présentation

**Projet :** ESIGELEC — AI with Python (Lab 5)  
**Équipe :** 3 personnes  
**Durée cible :** 5–8 minutes + démo live  

Ce document sert à préparer l’oral. Chacun lit sa partie à voix haute une fois avant demain.

---

## Ce qui est noté (d’après le lab05)

Le projet est évalué sur :

1. **Completeness** — idée → données → entraînement → démo
2. **Appropriate data handling** — nos images, CVAT, splits train/val
3. **Working code** — ça marche en live
4. **Quality of the final presentation** — clarté de l’explication

---

## Répartition des rôles (équipe de 3)

| Personne | Rôle | Slides / moments |
|----------|------|------------------|
| **Personne A** | Produit + pipeline | Slides 1, 2, 3 + intro |
| **Personne B** | Données + modèle | Slides 4, 5 |
| **Personne C** | Code + démo + conclusion | Slides 6, 7, 8, 9 + lance la démo |

Adaptez les noms (A/B/C) avec vos vrais prénoms.

---

## Commande de démo (à tester AVANT)

```bash
cd /Users/wadjoud/Ping-Pong
source .venv/bin/activate
python -m src.main --gestures --model models/best.pt --conf 0.25 --device mps
```

**Plan B** si `mps` ne marche pas :

```bash
python -m src.main --gestures --model models/best.pt --conf 0.25 --device cpu
```

**Plan B2** (détection seule, sans jeu) :

```bash
python scripts/webcam_detection.py --model models/best.pt --conf 0.25
```

Contrôles pendant la démo :
- **Poing** → raquette gauche (Joueur 1)
- **Main ouverte** → raquette droite (Joueur 2)
- Clavier de secours : `W/S` et flèches
- Quitter : `Q` ou `ESC`

---

## Chiffres à retenir (tout le monde)

- Classes : `fist` (poing) = Joueur 1 · `open_hand` (main ouverte) = Joueur 2
- Train : **216 images** → 156 fist + 60 open_hand
- Val : **54 images** → 26 fist + 28 open_hand
- Modèle : **YOLO11n fine-tuné**
- mAP validation ≈ **0.90**
- Perf : détection **1 frame sur 3**, `imgsz=320`, device **MPS**

---

## Script oral (à lire / apprendre)

### PERSONNE A

#### Slide 1 — Titre
> Bonjour, on est une équipe de trois.  
> Notre projet, c’est un **jeu de ping-pong contrôlé par gestes de mains**, avec la webcam et un modèle YOLO que **nous avons entraîné nous-mêmes**.

#### Slide 2 — Idée
> L’idée est simple : au lieu d’utiliser seulement le clavier, chaque joueur contrôle sa raquette avec un geste.  
> **Joueur 1** = le poing (`fist`).  
> **Joueur 2** = la main ouverte (`open_hand`).  
> La webcam détecte le geste, et le jeu réagit en direct.

#### Slide 3 — Pipeline
> Notre système a trois étapes :  
> 1. la **webcam** capture l’image,  
> 2. **YOLO** détecte `fist` ou `open_hand`,  
> 3. le **jeu** déplace la raquette selon la hauteur de la main.  
> Donc le pipeline est : **Webcam → YOLO → actions du jeu**.  
> Je passe la parole à [Personne B] pour les données et le modèle.

---

### PERSONNE B

#### Slide 4 — Dataset & annotation
> On n’a pas utilisé un dataset tout fait : on a **collecté nos propres images**.  
> On les a annotées avec **CVAT**, au format **YOLO Detection**.  
> Les classes sont exactement : `fist` et `open_hand`.  
>  
> Pour l’entraînement :  
> - **train** : 216 images (156 poings, 60 mains ouvertes)  
> - **val** : 54 images (26 poings, 28 mains ouvertes)  
>  
> Important : les sessions train et val sont **séparées**, pour que le modèle ne “triche” pas en mémorisant le même fond ou les mêmes conditions.

#### Slide 5 — Entraînement & résultats
> On est partis de **YOLO11n** déjà pré-entraîné, puis on a fait un **fine-tune** sur notre dataset.  
> Au premier essai, la main ouverte marchait bien, mais le **poing était faible** : parfois non détecté, parfois confondu avec `open_hand`.  
> On a donc ajouté **55 photos de poing** dans les conditions réelles de démo, on les a annotées, puis on a **réentraîné**.  
> Sur la validation, on obtient un mAP autour de **0.90**.  
> La main ouverte reste plus fiable, mais le poing est nettement mieux.  
> Je passe à [Personne C] pour le code et la démo.

---

### PERSONNE C

#### Slide 6 — Technique
> Côté technique, on utilise **Python**, **OpenCV**, **Ultralytics YOLO** et **Pygame**.  
> Le jeu reste jouable au clavier en secours.  
> Pour que ce soit fluide, on ne lance pas YOLO à chaque image : on détecte **1 frame sur 3**, et on réutilise le dernier geste entre deux détections.  
> Sur Mac, on utilise l’accélération **MPS**.

#### Slide 7 — Démo live
> On passe à la démonstration.  
> *(Lancer la commande, montrer poing puis main ouverte)*  
> Voici le **poing** → raquette gauche.  
> Voici la **main ouverte** → raquette droite.

#### Slide 8 — Limites & améliorations
> Quelques limites honnêtes :  
> - le poing reste plus difficile à détecter que la main ouverte,  
> - la détection coûte du temps, donc il faut optimiser,  
> - on n’a pas pu faire un vrai set `test` à cause de la limite de tasks sur CVAT gratuit.  
>  
> Améliorations possibles : plus de données poing variées, YOLO dans un thread séparé, et une interface de jeu plus soignée.

#### Slide 9 — Conclusion
> En résumé : on a fait le pipeline complet — collecte, annotation CVAT, fine-tune YOLO, et une démo live d’un jeu piloté par gestes.  
> Merci. On est prêts pour vos questions.

---

## Questions possibles (réponses courtes)

**Pourquoi train et val séparés ?**  
> Pour mesurer si le modèle généralise, pas juste s’il a mémorisé les mêmes conditions.

**Pourquoi YOLO Detection et pas Pose ?**  
> On a besoin de détecter des classes de gestes avec des boîtes. Pose serait utile si on voulait suivre des points précis de la main/doigts.

**Pourquoi le premier modèle ratait le poing ?**  
> Moins de variété / conditions différentes de la démo. On a ajouté des images de poing dans les conditions réelles, puis réentraîné.

**Le jeu est-il lent ?**  
> YOLO est lourd. On a réduit la fréquence de détection (1/3), baissé `imgsz`, et utilisé MPS.

---

## Checklist avant demain

- [ ] Les 3 ont lu leur partie à voix haute
- [ ] Démo testée au moins 1 fois sur la machine de présentation
- [ ] `models/best.pt` est bien présent
- [ ] Webcam autorisée / qui marche
- [ ] Qui lance la commande = Personne C (ou celui qui a le Mac)
- [ ] Slides prêtes (ou notes affichées)

---

## Message court à envoyer sur le groupe (optionnel)

> Salut, j’ai mis le script oral du projet dans `docs/presentation-orale.md`.  
> On se répartit : A = idée/pipeline, B = données/modèle, C = code/démo.  
> Demain on teste la démo avec :  
> `python -m src.main --gestures --model models/best.pt --conf 0.25 --device mps`  
> Lisez votre partie ce soir svp.
