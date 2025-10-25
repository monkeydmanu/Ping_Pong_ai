"""
Fonctions de dessin pour les objets du jeu.
"""

import pygame
import math
from config import RED, BROWN, GREEN, PINGPONG_ORANGE, WIDTH, HEIGHT, TABLE_Y

# Charger le fond UNE SEULE FOIS (au moment de l'import)
_background_image = pygame.image.load("graphics/assets/blue_background.png")
_background_image = pygame.transform.scale(_background_image, (WIDTH, HEIGHT))

def draw_background(screen):
    """Dessine l'image de fond."""
    screen.blit(_background_image, (0, 0))

def draw_table(screen, table):
    """Dessine la table à partir d'une instance Table."""
    x, y, w, h = table.get_rect()
    pygame.draw.rect(screen, BROWN, (x, y, w, h))

def draw_ball(screen, ball):
    """
    Dessine la balle avec un motif en spirale pour visualiser la rotation.
    - Le centre reste orange.
    - Plusieurs points blancs forment une spirale tournant avec ball.angle.
    """
    # créer une surface temporaire pour dessiner la balle
    surf = pygame.Surface((2 * ball.radius, 2 * ball.radius), pygame.SRCALPHA)
    
    # dessiner la balle de base
    pygame.draw.circle(surf, PINGPONG_ORANGE, (ball.radius, ball.radius), ball.radius)

    # paramètres du motif
    point_radius = max(2, ball.radius // 6)
    num_points = 6  # nombre de points dans la spirale
    spiral_tightness = 0.6  # contrôle à quel point la spirale est serrée

    # dessiner plusieurs points sur une spirale
    for i in range(num_points):
        # angle relatif du point (tourne avec ball.angle)
        theta = ball.angle + i * 0.8  # espacement angulaire des points
        # distance du centre (augmente progressivement)
        r = spiral_tightness * (i + 1) / num_points * ball.radius
        
        # coordonnées du point
        pos = pygame.math.Vector2(r, 0).rotate_rad(theta)
        px = ball.radius + pos.x
        py = ball.radius + pos.y
        
        pygame.draw.circle(surf, (255, 255, 255), (int(px), int(py)), point_radius)

    # placement final sur l’écran
    rect = surf.get_rect(center=ball.pos.astype(int))
    screen.blit(surf, rect.topleft)



def draw_paddle(screen, paddle, color):
    """
    Dessine une raquette sur l'écran en fonction de ses infos.
    paddle.get_info() doit retourner :
    center_x, center_y, vel_x, vel_y, angle, width, height
    """
    # Récupération des infos
    center_x, center_y, vel_x, vel_y, angle, width, height = paddle.get_info()
    
    # Création d'un rectangle centré
    rect = pygame.Rect(0, 0, width, height)
    rect.center = (center_x, center_y)
    
    # Surface temporaire pour rotation
    paddle_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    paddle_surf.fill(color)
    
    # Rotation de la surface
    angle = (angle)%360
    rotated_surf = pygame.transform.rotate(paddle_surf, angle)  # pygame tourne dans le sens antihoraire
    
    # Nouveau rectangle centré après rotation
    new_rect = rotated_surf.get_rect(center=(center_x, center_y))
    
    # Dessin sur l'écran
    screen.blit(rotated_surf, new_rect.topleft)

    

def draw_net(screen, net):
    """Dessine le filet blanc au centre de la table."""
    x, y, w, h = net.get_rect()
    pygame.draw.rect(screen, (255, 255, 255), (x, y, w, h))
