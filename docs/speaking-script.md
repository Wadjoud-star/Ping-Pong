# Speaking script — Gesture-controlled Ping-Pong

Team of 2. Only what to say.

---

## Person 1

### Slide 1 — Title
Hello. We are a team of two. Our project is a ping-pong game controlled by hand gestures, using a webcam and a YOLO model that we trained ourselves.

### Slide 2 — Idea
The idea is simple. Instead of using only the keyboard, each player controls a paddle with a gesture. Player 1 uses a fist. Player 2 uses an open hand. The webcam detects the gesture, and the game reacts in real time.

### Slide 3 — Pipeline
Our pipeline has three steps. First, the webcam captures the image. Second, YOLO detects fist or open_hand. Third, the game moves the paddle based on the vertical position of the hand. So the pipeline is: webcam, then YOLO, then game actions. I will now hand over to my teammate.

---

## Person 2

### Slide 4 — Dataset
We did not use a ready-made dataset. We collected our own images and annotated them with CVAT in YOLO Detection format. The classes are fist and open_hand. For training we used 216 images: 156 fists and 60 open hands. For validation we used 54 images: 26 fists and 28 open hands. Train and validation come from separate sessions, so the model does not just memorize the same background or the same conditions.

### Slide 5 — Training
We started from pretrained YOLO11n and fine-tuned it on our dataset. In the first training, open_hand worked well, but fist was weak. Sometimes it was not detected, and sometimes it was confused with open_hand. So we added 55 more fist images in real demo conditions, annotated them, and trained again. On validation, we get about 0.90 mAP. Open_hand is still more reliable, but fist is clearly better.

### Slide 6 — Technical mapping
On the technical side, we use Python, OpenCV, Ultralytics YOLO, and Pygame. Fist controls the left paddle. Open hand controls the right paddle. The keyboard still works as a backup. To keep the game smoother, we detect once every three frames and reuse the last gesture in between. On Mac, we use MPS.

### Slide 7 — Demo
We will now show the live demo. Fist controls the left paddle. Open hand controls the right paddle.

### Slide 8 — Limits and improvements
Some honest limitations: fist is still harder to detect than open_hand, detection can add latency, and we could not build a separate test set because of the free CVAT task limit. Possible improvements are more varied fist images, running YOLO in a separate thread, and a better game interface.

### Slide 9 — Conclusion
To summarize: we built the full pipeline — data collection, CVAT annotation, YOLO fine-tuning, and a live gesture-controlled game. Thank you. We are ready for your questions.
