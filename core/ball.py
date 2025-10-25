"""
Classe représentant la balle et sa physique.
"""

import numpy as np
from config import GRAVITY, BALL_RADIUS, FPS, PIXELS_PER_METER
from engine.collision import check_table_collision

class Ball:
    def __init__(self, x, y, vx, vy, radius=BALL_RADIUS, angular_speed=0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)      # pixels/sec
        self.radius = radius
        self.angle = 90      # angle actuel pour l'affichage
        self.angular_speed = angular_speed      # rad/s

    def update(self):
        """Met à jour la position et la vitesse de la balle."""
        accel = GRAVITY * PIXELS_PER_METER * 1/FPS
        self.vel[1] += GRAVITY  # Gravité
        # translation
        self.pos += self.vel * 1/60
        # rotation
        self.angle += self.angular_speed * 1/FPS
