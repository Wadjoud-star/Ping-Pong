"""Jeu ping-pong minimal (Pygame).

Contrôles clavier (toujours actifs) :
  Joueur 1 : W / S
  Joueur 2 : Flèches haut / bas
  R        : reset score + balle
  ESC / Q  : quitter

Contrôles gestes (optionnels, via ControlState) :
  Joueur 1 (fist)      → paddle gauche suit la position Y normalisée
  Joueur 2 (open_hand) → paddle droit suit la position Y normalisée
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pygame


WIDTH, HEIGHT = 960, 540
PADDLE_W, PADDLE_H = 14, 100
BALL_SIZE = 14
PADDLE_SPEED = 8
BALL_SPEED = 6
FPS = 60
WHITE = (240, 240, 240)
BG = (18, 22, 32)
ACCENT = (70, 200, 140)
P1_COLOR = (90, 170, 255)
P2_COLOR = (255, 140, 90)
MUTED = (120, 130, 150)


@dataclass
class ControlState:
    """Entrées externes (gestes YOLO). Y ∈ [0, 1] ; None = pas de commande geste."""

    p1_y: float | None = None
    p2_y: float | None = None
    p1_hit: bool = False
    p2_hit: bool = False
    labels: dict[str, str] = field(default_factory=dict)


class PongGame:
    def __init__(self, title: str = "Ping-Pong | Gestures") -> None:
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo", 28)
        self.small = pygame.font.SysFont("menlo", 16)
        self.controls = ControlState()
        self.running = True
        self._reset_round(serve_to_right=True)
        self.score = [0, 0]

    def _reset_round(self, serve_to_right: bool = True) -> None:
        self.p1 = pygame.Rect(40, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
        self.p2 = pygame.Rect(WIDTH - 40 - PADDLE_W, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.vx = BALL_SPEED if serve_to_right else -BALL_SPEED
        self.vy = BALL_SPEED * 0.6

    def set_controls(self, controls: ControlState) -> None:
        self.controls = controls

    def _move_paddle(self, paddle: pygame.Rect, dy: float) -> None:
        paddle.y = int(max(0, min(HEIGHT - paddle.height, paddle.y + dy)))

    def _set_paddle_norm_y(self, paddle: pygame.Rect, norm_y: float) -> None:
        """Place le centre du paddle à une hauteur normalisée [0, 1]."""
        norm_y = max(0.0, min(1.0, norm_y))
        center = norm_y * HEIGHT
        paddle.y = int(max(0, min(HEIGHT - paddle.height, center - paddle.height / 2)))

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.score = [0, 0]
                    self._reset_round(True)

    def update(self) -> None:
        keys = pygame.key.get_pressed()

        # Clavier
        if keys[pygame.K_w]:
            self._move_paddle(self.p1, -PADDLE_SPEED)
        if keys[pygame.K_s]:
            self._move_paddle(self.p1, PADDLE_SPEED)
        if keys[pygame.K_UP]:
            self._move_paddle(self.p2, -PADDLE_SPEED)
        if keys[pygame.K_DOWN]:
            self._move_paddle(self.p2, PADDLE_SPEED)

        # Gestes (écrasent la position si présents)
        if self.controls.p1_y is not None:
            self._set_paddle_norm_y(self.p1, self.controls.p1_y)
        if self.controls.p2_y is not None:
            self._set_paddle_norm_y(self.p2, self.controls.p2_y)

        # Boost ponctuel sur "hit" geste
        if self.controls.p1_hit and self.ball.colliderect(self.p1.inflate(20, 10)):
            self.vx = abs(self.vx) * 1.15
        if self.controls.p2_hit and self.ball.colliderect(self.p2.inflate(20, 10)):
            self.vx = -abs(self.vx) * 1.15

        self.ball.x += int(self.vx)
        self.ball.y += int(self.vy)

        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.vy = -self.vy
            self.ball.y = max(0, min(HEIGHT - self.ball.height, self.ball.y))

        if self.ball.colliderect(self.p1) and self.vx < 0:
            self.vx = abs(self.vx)
            offset = (self.ball.centery - self.p1.centery) / (PADDLE_H / 2)
            self.vy = BALL_SPEED * offset
        if self.ball.colliderect(self.p2) and self.vx > 0:
            self.vx = -abs(self.vx)
            offset = (self.ball.centery - self.p2.centery) / (PADDLE_H / 2)
            self.vy = BALL_SPEED * offset

        if self.ball.right < 0:
            self.score[1] += 1
            self._reset_round(serve_to_right=True)
        elif self.ball.left > WIDTH:
            self.score[0] += 1
            self._reset_round(serve_to_right=False)

        # Cap vitesse
        max_v = BALL_SPEED * 2.5
        self.vx = max(-max_v, min(max_v, self.vx))
        self.vy = max(-max_v, min(max_v, self.vy))

    def draw(self) -> None:
        self.screen.fill(BG)
        for y in range(0, HEIGHT, 24):
            pygame.draw.rect(self.screen, MUTED, (WIDTH // 2 - 2, y, 4, 12))

        pygame.draw.rect(self.screen, P1_COLOR, self.p1, border_radius=4)
        pygame.draw.rect(self.screen, P2_COLOR, self.p2, border_radius=4)
        pygame.draw.ellipse(self.screen, ACCENT, self.ball)

        score_surf = self.font.render(f"{self.score[0]}   {self.score[1]}", True, WHITE)
        self.screen.blit(score_surf, score_surf.get_rect(center=(WIDTH // 2, 36)))

        hint = "P1: W/S ou fist  |  P2: ↑/↓ ou open_hand  |  R reset  |  Q quit"
        hint_surf = self.small.render(hint, True, MUTED)
        self.screen.blit(hint_surf, (16, HEIGHT - 28))

        g1 = self.controls.labels.get("p1", "—")
        g2 = self.controls.labels.get("p2", "—")
        status = self.small.render(f"geste P1: {g1}   geste P2: {g2}", True, MUTED)
        self.screen.blit(status, (16, 16))

        pygame.display.flip()

    def tick(self) -> bool:
        """Une frame. Retourne False si le jeu doit s'arrêter."""
        self.handle_events()
        if not self.running:
            return False
        self.update()
        self.draw()
        self.clock.tick(FPS)
        return True

    def run_keyboard_only(self) -> None:
        while self.tick():
            pass
        pygame.quit()


def main() -> int:
    PongGame().run_keyboard_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
