"""
Classe représentant la balle et sa physique.
"""

import numpy as np
from config import GRAVITY, BALL_RADIUS, FPS
from engine.collision import check_table_collision

class Ball:
    def __init__(self, x, y, vx, vy, radius=BALL_RADIUS, angular_speed=0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)      # pixels/frame
        self.radius = radius
        self.angle = 90      # angle actuel pour l'affichage
        self.angular_speed = angular_speed      # rad/s
        self.collision_cooldown = 0  # frames restantes avant prochaine collision possible

    def update(self):
        """Met à jour la position et la vitesse de la balle avec effet Magnus et traînée."""
        dt = 1.0 / FPS  # Temps réel par frame (1/120 = 0.0083s)
        
        # Décrémenter le cooldown de collision
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        
        # Vitesse en pixels/s pour calculer la norme
        speed_px = np.linalg.norm(self.vel)
        
        if speed_px > 1.0:  # éviter division par zéro
            # === EFFET MAGNUS (simplifié pour le jeu) ===
            magnus_strength = 0.5  # ajustable pour plus/moins d'effet
            
            # Topspin (angular_speed > 0 avec vel[0] > 0) → force vers le bas (+y)
            if self.vel[0] != 0:
                magnus_accel_y = magnus_strength * self.angular_speed * np.sign(self.vel[0])
                self.vel[1] += magnus_accel_y * dt
            
            # Composante horizontale (plus faible)
            if self.vel[1] != 0:
                magnus_accel_x = -magnus_strength * 0.3 * self.angular_speed * np.sign(self.vel[1])
                self.vel[0] += magnus_accel_x * dt
            
            # === TRAÎNÉE AÉRODYNAMIQUE (légère) ===
            drag_factor = 0.5  # par seconde
            self.vel[0] *= (1 - drag_factor * dt)
            self.vel[1] *= (1 - drag_factor * dt)
        
        # Gravité : GRAVITY=9.81 m/s², converti en pixels/s²
        gravity_pixels = GRAVITY * 200  # ~2000 pixels/s²
        self.vel[1] += gravity_pixels * dt
        
        # Translation
        self.pos += self.vel * dt
        
        # Rotation visuelle
        self.angle += self.angular_speed * dt
        
        # Décroissance naturelle du spin
        self.angular_speed *= (1 - 0.1 * dt)  # ~10% par seconde


def spawn_ball_left(table):
    """Crée une balle au bord gauche de la table."""
    x_table, y_table, w_table, h_table = table.get_rect()
    return Ball(
        x=x_table + 30,  # Bord gauche de la table
        y=y_table - 300,  # Au-dessus de la table
        vx=0,
        vy=0,
        angular_speed=0
    )


def spawn_ball_right(table):
    """Crée une balle au bord droit de la table."""
    x_table, y_table, w_table, h_table = table.get_rect()
    return Ball(
        x=x_table + w_table - 30,  # Bord droit de la table
        y=y_table - 300,  # Au-dessus de la table
        vx=0,
        vy=0,
        angular_speed=0
    )
