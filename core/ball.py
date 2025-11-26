"""
Classe représentant la balle et sa physique.
"""

import numpy as np
from config import (
    GRAVITY, BALL_RADIUS, FPS, PIXELS_PER_METER,
    BALL_MASS, BALL_REAL_RADIUS, AIR_DENSITY,
    MAGNUS_COEFFICIENT, DRAG_COEFFICIENT
)
from engine.collision import check_table_collision

class Ball:
    def __init__(self, x, y, vx, vy, radius=BALL_RADIUS, angular_speed=0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)      # pixels/sec
        self.radius = radius
        self.angle = 90      # angle actuel pour l'affichage
        self.angular_speed = angular_speed      # rad/s

    def update(self):
        """Met à jour la position et la vitesse de la balle avec effet Magnus et traînée."""
        dt = 1.0 / FPS
        
        # Convertir la vitesse en m/s pour les calculs physiques
        vel_ms = self.vel / PIXELS_PER_METER
        speed = np.linalg.norm(vel_ms)
        
        if speed > 0.01:  # éviter division par zéro
            # Section transversale de la balle
            area = np.pi * BALL_REAL_RADIUS ** 2
            
            # === EFFET MAGNUS ===
            # Force de Magnus perpendiculaire à la vitesse
            # F_magnus = 0.5 * Cl * rho * A * v² * (direction perpendiculaire)
            # Le spin crée une force latérale (ici verticale en 2D)
            # Topspin (angular_speed > 0 avec vel[0] > 0) → force vers le bas
            # Backspin → force vers le haut
            magnus_magnitude = 0.5 * MAGNUS_COEFFICIENT * AIR_DENSITY * area * speed ** 2
            
            # Direction de la force Magnus : perpendiculaire à la vitesse
            # En 2D, si la balle va vers la droite et a du topspin, elle plonge
            # Signe basé sur : omega × v (produit vectoriel en 2D)
            # angular_speed positif + vel[0] positif → force vers le bas (+y)
            magnus_direction_y = np.sign(self.angular_speed) * np.sign(vel_ms[0]) if vel_ms[0] != 0 else 0
            magnus_force_y = magnus_magnitude * magnus_direction_y
            
            # Aussi une composante horizontale (moins importante)
            magnus_direction_x = -np.sign(self.angular_speed) * np.sign(vel_ms[1]) if vel_ms[1] != 0 else 0
            magnus_force_x = magnus_magnitude * 0.3 * magnus_direction_x  # facteur réduit
            
            # === TRAÎNÉE AÉRODYNAMIQUE ===
            # F_drag = 0.5 * Cd * rho * A * v² (opposée à la vitesse)
            drag_magnitude = 0.5 * DRAG_COEFFICIENT * AIR_DENSITY * area * speed ** 2
            drag_force_x = -drag_magnitude * (vel_ms[0] / speed)
            drag_force_y = -drag_magnitude * (vel_ms[1] / speed)
            
            # Accélérations (F = ma)
            accel_magnus_x = magnus_force_x / BALL_MASS
            accel_magnus_y = magnus_force_y / BALL_MASS
            accel_drag_x = drag_force_x / BALL_MASS
            accel_drag_y = drag_force_y / BALL_MASS
            
            # Appliquer les accélérations (convertir en pixels/s²)
            self.vel[0] += (accel_magnus_x + accel_drag_x) * PIXELS_PER_METER * dt
            self.vel[1] += (accel_magnus_y + accel_drag_y) * PIXELS_PER_METER * dt
        
        # Gravité (toujours appliquée)
        self.vel[1] += GRAVITY * PIXELS_PER_METER * dt
        
        # Translation
        self.pos += self.vel * dt
        
        # Rotation visuelle
        self.angle += self.angular_speed * dt
        
        # Décroissance naturelle du spin (friction de l'air sur la rotation)
        self.angular_speed *= 0.999  # très léger amortissement
