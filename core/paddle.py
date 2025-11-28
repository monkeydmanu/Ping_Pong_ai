#si j'appuis sur un bouton alors je peux prendre le contrôle de la raquette de mon choix

"""
Classe représentant une raquette
"""

import numpy as np
from config import HEIGHT, WIDTH, RACKET_HEIGHT_PX, RACKET_WIDTH_PX, SPEED_RACKET

class Paddle:
    def __init__(self, x, y, width=RACKET_WIDTH_PX, height=RACKET_HEIGHT_PX, speed=SPEED_RACKET, max_speed=None, x_min=0, x_max=WIDTH):
        self.pos = np.array([x, y], dtype=float)  # Position
        self.width = width
        self.height = height
        self.speed = speed  # Vitesse maximale de déplacement
        self.max_speed = max_speed if max_speed else speed  # Vitesse max = speed par défaut
        self.vel = np.array([0.0, 0.0], dtype=float)  # vélocité [vx, vy]
        self.angle = 0  # Rotation libre de la raquette
        self.acceleration = speed * 30  # Accélération très rapide (quasi instantanée)
        self.friction = 15  # Friction pour ralentir quand on lâche
        self.x_min = x_min  # Limite gauche
        self.x_max = x_max  # Limite droite

    # Mise à jour de la position selon la vélocité et le dt
    def update(self, dt):
        # Limite la vitesse au max
        speed_magnitude = np.linalg.norm(self.vel)
        if speed_magnitude > self.max_speed:
            self.vel = self.vel / speed_magnitude * self.max_speed
        
        self.pos += self.vel * dt

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
    def move_up(self):
        self.vel[1] -= self.acceleration * (1.0 / 120.0)  # accélère vers le haut

    def move_down(self):
        self.vel[1] += self.acceleration * (1.0 / 120.0)  # accélère vers le bas

    def stop_vertical(self):
        self.vel[1] = 0  # Arrêt instantané

    # Mouvement horizontal avec accélération
    def move_left(self):
        self.vel[0] -= self.acceleration * (1.0 / 120.0)

    def move_right(self):
        self.vel[0] += self.acceleration * (1.0 / 120.0)

    def stop_horizontal(self):
        self.vel[0] = 0  # Arrêt instantané

    # Rotation, sens trigo
    def rotate_left(self, dt, rotation_speed=6):
        self.angle -= rotation_speed * dt
        self.angle %= 360

    def rotate_right(self, dt, rotation_speed=6):
        self.angle += rotation_speed * dt
        self.angle %= 360

    # Retourne les infos essentielles pour collision/affichage
    def get_info(self):
        center_x = self.pos[0] + self.width / 2
        center_y = self.pos[1] + self.height / 2
        return center_x, center_y, self.vel[0], self.vel[1], self.angle, self.width, self.height
    
    def get_rect(self):
        return (self.pos[0], self.pos[1], self.width, self.height)