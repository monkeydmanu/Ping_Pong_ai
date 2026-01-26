#si j'appuis sur un bouton alors je peux prendre le contrôle de la raquette de mon choix

"""
Classe représentant une raquette
"""

import numpy as np
from config import (
    HEIGHT,
    WIDTH,
    RACKET_HEIGHT_PX,
    RACKET_WIDTH_PX,
    SPEED_RACKET,
    FPS,
    FPS_BASE,
    PADDLE_SPEED_SCALE,
    PADDLE_ROT_SCALE,
)

class Paddle:
    def __init__(self, x, y, width=RACKET_WIDTH_PX, height=RACKET_HEIGHT_PX, speed=SPEED_RACKET, max_speed=None, x_min=0, x_max=WIDTH):
        self.pos = np.array([x, y], dtype=float)  # Position
        self.width = width
        self.height = height

        effective_speed = speed * PADDLE_SPEED_SCALE
        self.speed = effective_speed  # Vitesse maximale de déplacement (scalée)
        self.max_speed = (max_speed * PADDLE_SPEED_SCALE) if max_speed else effective_speed
        self.vel = np.array([0.0, 0.0], dtype=float)  # vélocité [vx, vy]
        self.angle = 0  # Rotation libre de la raquette
        self.acceleration = effective_speed * 30  # Accélération rapide, cohérente avec la vitesse scalée
        self.friction = 15  # Friction pour ralentir quand on lâche
        self.x_min = x_min  # Limite gauche
        self.x_max = x_max  # Limite droite
        self.can_hit = True  # Booléen pour savoir si la raquette peut toucher la balle

    # Mise à jour de la position selon la vélocité et le dt
    def update(self, dt, speed_factor=1.0):
        """Met à jour la position de la raquette.
        
        Args:
            dt: Temps écoulé
            speed_factor: Facteur de curriculum (0.5 = lent, 1.0 = normal)
        """
        # Limiter la vitesse au max
        speed_magnitude = np.linalg.norm(self.vel)
        if speed_magnitude > self.max_speed:
            self.vel = self.vel / speed_magnitude * self.max_speed
        
        # Appliquer speed_factor à la vélocité pour curriculum
        self.pos += self.vel * dt * speed_factor

        # Limite verticale pour ne pas sortir de l'écran
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] = 0
        if self.pos[1] + self.height > HEIGHT:
            self.pos[1] = HEIGHT - self.height
            self.vel[1] = 0

        # Limite horizontale avec x_min et x_max
        if self.pos[0] < self.x_min:
            self.pos[0] = self.x_min
            self.vel[0] = 0
        if self.pos[0] + self.width > self.x_max:
            self.pos[0] = self.x_max - self.width
            self.vel[0] = 0

    # Mouvement vertical avec accélération
    def move_up(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[1] -= self.acceleration * dt  # accélère vers le haut

    def move_down(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[1] += self.acceleration * dt  # accélère vers le bas

    def stop_vertical(self):
        self.vel[1] = 0  # Arrêt instantané

    # Mouvement horizontal avec accélération
    def move_left(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[0] -= self.acceleration * dt

    def move_right(self, dt=None):
        if dt is None:
            dt = 1.0 / FPS
        self.vel[0] += self.acceleration * dt

    def stop_horizontal(self):
        self.vel[0] = 0  # Arrêt instantané

    # Rotation, sens trigo
    def rotate_left(self, dt, rotation_speed=6):
        self.angle -= (rotation_speed * PADDLE_ROT_SCALE) * dt
        self.angle %= 360

    def rotate_right(self, dt, rotation_speed=6):
        self.angle += (rotation_speed * PADDLE_ROT_SCALE) * dt
        self.angle %= 360

    # Retourne les infos essentielles pour collision/affichage
    def get_info(self):
        center_x = self.pos[0] + self.width / 2
        center_y = self.pos[1] + self.height / 2
        return center_x, center_y, self.vel[0], self.vel[1], self.angle, self.width, self.height
    
    def get_rect(self):
        return (self.pos[0], self.pos[1], self.width, self.height)