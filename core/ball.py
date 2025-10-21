"""
Classe représentant la balle et sa physique.
"""

import numpy as np
from config import GRAVITY, DT
from engine.collision import check_table_collision

class Ball:
    def __init__(self, x, y, vx, vy, radius=10, angular_speed=0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)      # pixels/sec
        self.radius = radius
        self.angle = 0      # angle actuel pour l'affichage
        self.angular_speed = angular_speed      # rad/s

    def update(self):
        """Met à jour la position et la vitesse de la balle."""
        self.vel[1] += GRAVITY * DT * 50  # Gravité (×50 pour être visible)
        # translation
        self.pos += self.vel * DT
        # rotation
        self.angle += self.angular_speed * DT
