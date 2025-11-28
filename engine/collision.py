"""
Fonctions de collision balle ↔ raquette / filet / table
"""

import pygame
import numpy as np
from config import RESTITUTION, TABLE_Y, VPX_FRAME_MAX

# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_verticale(vy, v0, a, ey_min=0.85):
    """
    Calcule ey (coeff. de restitution vertical) en fonction de la vitesse verticale vy.

    Params:
      vy      : float, vitesse verticale (m/s) — signe ignoré (on utilise |vy|)
      v0      : float, vitesse de référence (m/s) — pour laquelle ey doit atteindre ey_min
      a       : float, exposant contrôlant la pente (plus a grand → moins de restitution)
      ey_min  : float, valeur minimale de ey (par défaut 0.85)

    Retour :
      ey : float dans [ey_min, 1]
    """
    # normalisation et clamp
    x = abs(vy) / v0
    if x >= 1.0:
        return float(ey_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ey = ey_min + (1.0 - ey_min) * (1.0 - x) ** a

    # sécurité numérique
    if ey < ey_min:
        ey = ey_min
    if ey > 1.0:
        ey = 1.0
    return float(ey)


# pour le moment on prend la même que y
# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_tangentielle(vx, v0, a, ex_min=0.85):
    """
    Calcule le coefficient de restitution tangentielle ex selon la formule empirique :
        ex = 1 - (vx / vx0)**a

    Paramètres :
    ------------
    vx : float
        Vitesse tangentielle (horizontale) avant impact (m/s)
    vx0 : float
        Vitesse caractéristique liée à la friction (m/s)
    a : float
        Exposant empirique lié au type de surface

    Retour :
    --------
    ex : float
        Coefficient de restitution tangentielle (sans unité)
    """

    # normalisation et clamp
    z = abs(vx) / v0
    if z >= 1.0:
        return float(ex_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ex = ex_min + (1.0 - ex_min) * (1.0 - z) ** a

    # sécurité numérique
    if ex < ex_min:
        ex = ex_min
    if ex > 1.0:
        ex = 1.0
    return ex


def adjust_spin_for_corner(angular_speed, ratio, is_left_corner=True):
    """
    Ajuste la vitesse angulaire selon le coin et le ratio.
    - Si la spin est déjà dans le bon sens → augmente avec ratio
    - Sinon → tend vers le bon signe progressivement
    """
    # Coin gauche : bon sens = négatif
    if is_left_corner:
        if angular_speed < 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) - 0.8 * ratio * abs(angular_speed)  # tend vers négatif
    # Coin droit : bon sens = positif
    else:
        if angular_speed > 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) + 0.8 * ratio * abs(angular_speed)  # tend vers positif

def reduction_speed(vx, vy, est_mousse):
    if est_mousse:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX, 0.35)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX, 0.35)
    else:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX*1.2, 0.22)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX*1.2, 0.22)
    return vx, vy

def contact_cercle_rectangle(ball_center_x, ball_center_y, radius,
                             rect_x, rect_y, width, height, angle_deg, screen=None):
    """
    Retourne (hit, contact_world, normal_world, tangent_world, face)
    - hit: bool collision
    - contact_world: (x,y) point de contact (approx) en coordonnées monde
    - normal_world: (nx, ny) normale unitaire pointant vers l'extérieur (depuis la surface)
    - tangent_world: (tx, ty) vecteur tangent unitaire (direction de la surface)
    - face: 'haut' / 'bas' / 'gauche' / 'droite' / 'inside' / 'corner'
    Note: angle_deg est l'angle Pygame (0° = haut, anti-horaire).
    """

    # 0) Setup
    half_w = width / 2.0
    half_h = height / 2.0
    rect_cx = rect_x + half_w   # centre du rectangle (x)
    rect_cy = rect_y + half_h   # centre du rectangle (y)

    # 1) conversion angle Pygame -> angle trigonométrique (0° = droite)
    theta = np.radians(angle_deg)

    # 2) translation balle -> repère centré sur rectangle
    dx = ball_center_x - rect_cx
    dy = ball_center_y - rect_cy

    # 3) rotation inverse pour ramener dans repère non-roté du rectangle
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    rx = dx * cos_t - dy * sin_t
    ry = dx * sin_t + dy * cos_t

    # 4) trouver le point le plus proche sur le rectangle axis-aligned
    closest_x = np.clip(rx, -half_w, half_w)
    closest_y = np.clip(ry, -half_h, half_h)

    # 5) distance au point le plus proche
    dist_x = rx - closest_x
    dist_y = ry - closest_y
    distance_sq = dist_x*dist_x + dist_y*dist_y
    hit = distance_sq <= radius*radius

    # valeurs par défaut si pas de collision
    contact_world = None
    normal_world = None
    tangent_world = None
    face = None
    corner_ratio = None


    # c'était ici le if not hit return


    # 6) point de contact en coordonnées locales (repère non-roté)
    contact_local_x = closest_x
    contact_local_y = closest_y

    # 7) classification : quel axe a été clampé ?
    outside_x = (rx < -half_w) or (rx > half_w)
    outside_y = (ry < -half_h) or (ry > half_h)

    if outside_x and not outside_y:
        # collision avec un bord vertical (gauche or droite)
        face = 'gauche' if rx < -half_w else 'droite'
        # normale locale pointe vers l'extérieur du rectangle
        n_local = (-1.0, 0.0) if rx < -half_w else (1.0, 0.0)
    elif outside_y and not outside_x:
        # collision avec un bord horizontal (haut or bas)
        face = 'haut' if ry < -half_h else 'bas'
        n_local = (0.0, -1.0) if ry < -half_h else (0.0, 1.0)
    elif (not outside_x) and (not outside_y):
        # le centre du cercle est au dessus de la surface (proche d'une face interne)
        # ici on est collé à l'intérieur de la projection : considérer la face la plus proche
        # calculer les distances aux 4 bords et choisir la plus petite
        dist_to_right = half_w - rx
        dist_to_left  = rx + half_w
        dist_to_top   = ry + half_h
        dist_to_bottom= half_h - ry
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        if min_dist == dist_to_left:
            face = 'gauche'; n_local = (-1.0, 0.0)
        elif min_dist == dist_to_right:
            face = 'droite'; n_local = (1.0, 0.0)
        elif min_dist == dist_to_top:
            face = 'haut'; n_local = (0.0, -1.0)
        else:
            face = 'bas'; n_local = (0.0, 1.0)
    else:
        # Coin : déterminer lequel
        left  = rx < -half_w
        right = rx >  half_w
        top   = ry < -half_h
        bottom= ry >  half_h
        if left and top:
            face = 'corner_hg'
            dist_edge_x = (-half_w) - rx  # positif
        elif right and top:
            face = 'corner_hd'
            dist_edge_x = rx - half_w
        elif left and bottom:
            face = 'corner_bg'
            dist_edge_x = (-half_w) - rx
        else:
            face = 'corner_bd'
            dist_edge_x = rx - half_w
        # ratio basé sur distance horizontale
        corner_ratio = min(1.0, abs(dist_edge_x) / radius)
        nx_local = rx - closest_x
        ny_local = ry - closest_y
        length = np.hypot(nx_local, ny_local)
        n_local = (0.0, 0.0) if length == 0 else (nx_local/length, ny_local/length)

    # 8) projeter contact_local -> monde (rotation + translation)
    cos_t_forward = np.cos(theta)
    sin_t_forward = np.sin(theta)
    contact_world_x = contact_local_x * cos_t_forward - contact_local_y * sin_t_forward + rect_cx
    contact_world_y = contact_local_x * sin_t_forward + contact_local_y * cos_t_forward + rect_cy
    contact_world = (contact_world_x, contact_world_y)

    # 9) normale locale -> normale monde (rotation)
    nx_world = n_local[0] * cos_t_forward - n_local[1] * sin_t_forward
    ny_world = n_local[0] * sin_t_forward + n_local[1] * cos_t_forward

    # normaliser par sécurité la normale pour qu'elle est une norme de 1
    norm = np.hypot(nx_world, ny_world)
    if norm != 0:
        nx_world /= norm; ny_world /= norm
    normal_world = (nx_world, ny_world)

    # 10) tangente = rot90(normal) (unité)
    tx_world, ty_world = -ny_world, nx_world
    tangent_world = (tx_world, ty_world)

    # 11) Affichage Pygame du rectangle (après rotation)
    if screen is not None:
    # Coins locaux du rectangle
        corners_local = [
            (-half_w, -half_h),
            ( half_w, -half_h),
            ( half_w,  half_h),
            (-half_w,  half_h)
        ]
        corners_world = []
        for x, y in corners_local:
            wx = x * cos_t_forward - y * sin_t_forward + rect_cx
            wy = x * sin_t_forward + y * cos_t_forward + rect_cy
            corners_world.append((int(wx), int(wy)))
        
        # Tracer le rectangle et son centre toujours
        pygame.draw.polygon(screen, (0, 255, 0), corners_world, 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(rect_cx), int(rect_cy)), 4)

        # Tracer la balle toujours
        pygame.draw.circle(screen, (255, 255, 0), (int(ball_center_x), int(ball_center_y)), int(radius), 1)

        # Tracer le point de contact uniquement si collision
        if hit:
            pygame.draw.circle(screen, (0, 0, 255),
                            (int(contact_world[0]), int(contact_world[1])), 4)
            
    
    if not hit:
        return False, contact_world, normal_world, tangent_world, face, corner_ratio

    return True, contact_world, normal_world, tangent_world, face, corner_ratio

# a=0.35 pour la mousse et 0.22 pour la table
def check_rect_collision(ball, rectangle, est_mousse, est_table, a, spin_factor=0.2, screen=None):
    """
    Gestion du rebond sur un rectangle incluant coins gauche/droite avec :
    - ratio basé sur le bord pour un effet plus marqué
    - spin ajusté en fonction du ratio et de la vitesse de base
    """
    ball_center_x = ball.pos[0]
    ball_center_y = ball.pos[1]

    signe_x = (ball.angular_speed) >= 0 # True si spin positif
    signe_y = (ball.angular_speed * ball.vel[0] >= 0)

    if est_mousse:
        _, _, vel_x, vel_y, angle, width, height = rectangle.get_info()
        x, y, _, _ = rectangle.get_rect()
    elif est_table:
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 # on ajoutera 90 après
    else: # filet
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 # on ajoutera 90 après

    hit, contact, normal, tangent, face, corner_ratio = contact_cercle_rectangle(
        ball_center_x, ball_center_y, ball.radius,
        x, y, width, height, angle, screen
    )

    if not hit:
        return

    # Éviter les collisions multiples avec la raquette (cooldown de 0.2s = 24 frames à 120fps)
    if est_mousse and ball.collision_cooldown > 0:
        return

    # Repositionnement pour éviter l'enfoncement
    ball.pos[0] = contact[0] + normal[0] * ball.radius
    ball.pos[1] = contact[1] + normal[1] * ball.radius
    
    # Activer le cooldown pour la raquette
    if est_mousse:
        ball.collision_cooldown = 24  # 0.2 secondes à 120fps

    if face and face.startswith('corner_'):
        # Ancienne logique coin
        ratio = corner_ratio if corner_ratio is not None else 0.0
        transfer = ratio * ball.vel[1]

        # Direction horizontale
        if 'hg' in face or 'bg' in face:  # gauche
            ball.vel[0] = -abs(ball.vel[0]) - abs(transfer)
            is_left_corner = True
        else:  # droite
            ball.vel[0] = abs(ball.vel[0]) + abs(transfer)
            is_left_corner = False

        # Direction verticale
        if 'hg' in face or 'hd' in face:  # haut
            ball.vel[1] = -abs(ball.vel[1]) * (1 - 0.5 * ratio)
        else:  # bas
            ball.vel[1] = abs(ball.vel[1]) * (1 - 0.5 * ratio)

        # Ajustement spin
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=is_left_corner)

        # Réduction vitesses
        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], est_mousse)

    else:
        # appliquer la réflexion selon la normale
        v_dot_n = ball.vel[0]*normal[0] + ball.vel[1]*normal[1]
        ball.vel[0] -= 2 * v_dot_n * normal[0]
        ball.vel[1] -= 2 * v_dot_n * normal[1]

        # === TRANSFERT VITESSE RAQUETTE → BALLE (seulement pour la mousse/raquette) ===
        if est_mousse:
            # Facteur de transfert de vitesse
            velocity_transfer = 0.7  # 70% de la vitesse transférée
            
            # Ajouter la vitesse de la raquette à la balle
            ball.vel[0] += vel_x * velocity_transfer
            ball.vel[1] += vel_y * velocity_transfer
            
            # === GÉNÉRATION DE SPIN BASÉE SUR OÙ ON TAPE LA BALLE ===
            # Vecteur du centre de la balle vers le point de contact
            contact_offset_x = contact[0] - ball_center_x
            contact_offset_y = contact[1] - ball_center_y
            
            # Normaliser par le rayon pour avoir un ratio [-1, 1]
            # contact_ratio_y > 0 = on tape en dessous de la balle (backspin/coupé)
            # contact_ratio_y < 0 = on tape au dessus de la balle (topspin)
            contact_ratio_y = contact_offset_y / ball.radius if ball.radius > 0 else 0
            
            # La vitesse horizontale de la raquette amplifie l'effet
            # Plus on frappe fort horizontalement, plus l'effet est marqué
            paddle_speed = np.sqrt(vel_x**2 + vel_y**2)
            
            # Générer le spin :
            # - Taper dessus (contact_ratio_y < 0) + mouvement vers la droite (vel_x > 0) = topspin (positif)
            # - Taper dessous (contact_ratio_y > 0) + mouvement vers la droite (vel_x > 0) = backspin (négatif)
            # Le signe du spin dépend de : -contact_ratio_y * signe(vel_x)
            spin_generation = 0.15  # facteur de conversion
            direction_x = 1 if vel_x >= 0 else -1  # direction du coup
            generated_spin = -contact_ratio_y * paddle_speed * spin_generation * direction_x
            
            ball.angular_speed += generated_spin

        # Ratio basé sur le spin (existant)
        max_spin = 500.0
        ratio = min(1.0, abs(ball.angular_speed) / max_spin)

        # Redistribution de l'énergie
        ball.vel[0] += abs(ball.angular_speed) * spin_factor * (ratio if signe_x else -ratio)
        ball.vel[1] += abs(ball.angular_speed) * spin_factor * (ratio) * (1 if signe_y else -1) # si dans le même sens alors on descend (positif) sinon on monte (negatif)

        ball.angular_speed *= 0.8

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], est_mousse)





def check_table_collision(ball, table):
    check_rect_collision(ball, table, est_mousse=False, est_table=True, a=0.22)



def check_ball_paddle(ball, paddle, screen):
    check_rect_collision(ball, paddle, est_mousse=True, est_table=False, a=0.35, screen=screen)

def check_ball_net(ball, net, restitution=RESTITUTION, spin_factor=0.3, spin_damping=0.8):
    """
    Gestion de la collision avec le filet :
    - ball : instance de Ball
    - net : instance de Net
    - restitution : coefficient pour vel_x inversé
    - spin_factor : influence de l'angular_speed sur vel_y
    - spin_damping : perte de spin après collision
    """
    check_rect_collision(ball, net, est_mousse=False, est_table=False, a=0.22)


