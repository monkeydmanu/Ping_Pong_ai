#si j'appuis sur un bouton alors je peux prendre le contrôle de la raquette de mon choix

"""
Classe représentant une raquette
"""

import numpy as np
from config import HEIGHT, WIDTH

class Paddle:
    def __init__(self, x, y, width=10, height=60, speed=10, max_speed=500):
        self.pos = np.array([x, y], dtype=float)  # Position
        self.width = width
        self.height = height
        self.speed = speed  # Vitesse de déplacement par défaut
        self.max_speed = max_speed  # Vitesse maximale en x ou y
        self.vel = np.array([0.0, 0.0], dtype=float)  # vélocité [vx, vy]
        self.angle = 0  # Rotation libre de la raquette

    # Mise à jour de la position selon la vélocité et le dt
    def update(self, dt):
        # Limite la vitesse
        self.vel = np.clip(self.vel, -self.max_speed, self.max_speed)
        self.pos += self.vel * dt

        # Limite verticale pour ne pas sortir de l'écran
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] = 0
        if self.pos[1] + self.height > HEIGHT:
            self.pos[1] = HEIGHT - self.height
            self.vel[1] = 0

        # Limite horizontale si nécessaire (optionnel)
        if self.pos[0] < 0:
            self.pos[0] = 0
            self.vel[0] = 0
        if self.pos[0] + self.width > WIDTH:
            self.pos[0] = WIDTH - self.width
            self.vel[0] = 0

    # Mouvement vertical direct (ex: touches)
    def move_up(self):
        self.vel[1] = -self.speed

    def move_down(self):
        self.vel[1] = self.speed

    def stop_vertical(self):
        self.vel[1] = 0

    # Mouvement horizontal direct (pour futur usage)
    def move_left(self):
        self.vel[0] = -self.speed

    def move_right(self):
        self.vel[0] = self.speed

    def stop_horizontal(self):
        self.vel[0] = 0

    # Rotation
    def rotate_left(self, dt, rotation_speed=6):
        self.angle -= rotation_speed * dt

    def rotate_right(self, dt, rotation_speed=6):
        self.angle += rotation_speed * dt

    # Retourne les infos essentielles pour collision/affichage
    def get_info(self):
        center_x = self.pos[0] + self.width / 2
        center_y = self.pos[1] + self.height / 2
        return center_x, center_y, self.vel[0], self.vel[1], self.angle, self.width, self.height
    
    def get_rect(self):
        return (self.pos[0], self.pos[1], self.width, self.height)