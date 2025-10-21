"""
Fonctions de collision balle ↔ raquette / filet / table
"""

from config import RESTITUTION, TABLE_Y

# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_verticale(v, v0, a):
    """
    Calcule le coefficient de restitution vertical ey selon la formule empirique :
        ey = 1 - (v / v0)**a

    Parameters:
    -----------
    v : float
        Vitesse verticale de la balle avant impact (m/s)
    v0 : float
        Vitesse caractéristique du matériau (m/s)
    a : float
        Exposant empirique lié au matériau (dimensionless)

    Returns:
    --------
    ey : float
        Coefficient de restitution vertical
    """
    ey = 1 - (v / v0) ** a
    # On s'assure que ey reste positif
    ey = max(0.0, ey)
    return ey


# k qui vaut 0.2 pour une balle en mousse disons, 0.08 pour une table rigide, R peut valoir 0.02 pour 2 cm
def restitution_spin(R, vitesse_angulaire_ini, v_x, k):
    """
    Calcule la vitesse angulaire après impact selon la vitesse horizontale et le coefficient k.
    
    Paramètres:
    -----------
    R : float
        Rayon de la balle (m)
    vitesse_angulaire_ini : float
        Vitesse angulaire avant impact (rad/s)
    v_x : float
        Vitesse horizontale de la balle avant impact (m/s)
    k : float
        Coefficient de conversion friction → rotation (adimensionnel)
        
    Retour:
    -------
    vitesse_angulaire_fin : float
        Vitesse angulaire après impact (rad/s)
    """
    vitesse_angulaire_fin = vitesse_angulaire_ini + k * v_x / R
    return vitesse_angulaire_fin


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

def reduction_speed(vx, vy, restitution, est_mousse):
    if est_mousse:
        vx *= restitution
        vy *= restitution_verticale(vy, 200, 0.35)
    else:
        vx *= restitution
        vy *= restitution_verticale(vy, 250, 0.22)
    return vx, vy

def check_rect_collision(ball, rectangle, est_mousse, est_table,restitution=RESTITUTION, spin_factor=0.2):
    """
    Gestion du rebond sur un rectangle incluant coins gauche/droite avec :
    - ratio basé sur le bord pour un effet plus marqué
    - spin ajusté en fonction du ratio et de la vitesse de base
    """
    x, y, w, h = rectangle.get_rect()
    ball_center_x = ball.pos[0]
    ball_center_y = ball.pos[1]

    # Cas 1 : collision classique sur la surface au dessus
    if y - ball.radius < ball_center_y <= y and x <= ball_center_x <= x + w:
        ball.pos[1] = y - ball.radius

        # Ratio basé sur le spin
        max_spin = 500.0
        ratio = min(1.0, abs(ball.angular_speed) / max_spin)

        print("rebond")
        # Déterminer le signe pour savoir si le spin est dans le bon sens
        print(ball.vel[0], ball.vel[1], ball.angular_speed)
        signe = abs(ball.angular_speed) / ball.angular_speed >= 0 # True si spin et vel_x même signe
        print(ball.angular_speed * spin_factor)
        print((ratio if signe else -ratio))

        # Redistribution de l'énergie
        ball.vel[0] = ball.vel[0] + abs(ball.angular_speed) * spin_factor * (ratio if signe else -ratio)
        ball.vel[1] = -ball.vel[1] + abs(ball.angular_speed) * spin_factor * (1 - 0.8 * ratio) * (1 if ball.angular_speed > 0 else -1) # si angular > 0 alors on enleve en y et sinon ça va encore plus haut

        print(ball.vel[0])
        # Atténuation du spin
        ball.angular_speed *= 0.8

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution, est_mousse, est_table)

    # Cas 2 : collision classique sur la surface mais en dessous
    if y + h < ball_center_y <= y + h + ball.radius and x <= ball_center_x <= x + w:
        print('cas 2')
        ball.pos[1] = y + h + ball.radius
        ball.vel[0] += spin_factor * ball.angular_speed
        ball.angular_speed *= 0.8

    # Cas 3 : coin haut gauche
    elif y - ball.radius < ball_center_y <= y and x - ball.radius < ball_center_x < x:
        print('cas 3')
        ball.pos[1] = y - ball.radius
        ball.pos[0] = x - ball.radius

        # Ratio depuis le bord
        ratio = min(1.0, abs(x - ball_center_x) / ball.radius)  # 0 = au bord, 1 = loin du bord
        # Transfert vertical -> horizontal
        transfer = ratio * ball.vel[1]
        ball.vel[0] = -abs(ball.vel[0]) - abs(transfer)  # vers la gauche
        ball.vel[1] = -abs(ball.vel[1]) * (1 - 0.5 * ratio) # vers le haut

        # Ajustement spin selon ratio : tend vers négatif si ratio grand
        # Coin gauche
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=True)
        
        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution, est_mousse, est_table)

    # Cas 4 : coin haut droit
    elif y - ball.radius < ball_center_y <= y and x + w + ball.radius > ball_center_x > x + w:
        print('cas 4')
        ball.pos[1] = y - ball.radius
        ball.pos[0] = x + w + ball.radius

        ratio = min(1.0, abs(ball_center_x - (x + w)) / ball.radius)
        transfer = ratio * ball.vel[1]
        ball.vel[0] = abs(ball.vel[0]) + abs(transfer)  # vers la droite
        ball.vel[1] = -abs(ball.vel[1]) * (1 - 0.5 * ratio) # vers le haut

        # Ajustement spin selon ratio : tend vers positif si ratio grand
        # Coin droit
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=False)

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution, est_mousse, est_table)

    # Cas 5 : coin bas gauche
    elif y + h < ball_center_y <= y + h + ball.radius and x - ball.radius < ball_center_x < x:
        print('cas 5')
        ball.pos[1] = y + h + ball.radius
        ball.pos[0] = x - ball.radius

        # Ratio depuis le bord
        ratio = min(1.0, abs(x - ball_center_x) / ball.radius)  # 0 = au bord, 1 = loin du bord
        # Transfert vertical -> horizontal
        transfer = ratio * ball.vel[1]
        ball.vel[0] = -abs(ball.vel[0]) - abs(transfer)  # vers la gauche
        ball.vel[1] = abs(ball.vel[1]) * (1 - 0.5 * ratio) # vers le bas

        # Ajustement spin selon ratio : tend vers négatif si ratio grand
        # Coin gauche
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=True)

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution, est_mousse, est_table)

    # Cas 6 : coin bas droit
    elif y + h < ball_center_y <= y + h + ball.radius and x + w + ball.radius > ball_center_x > x + w:
        print('cas 6')
        ball.pos[1] = y + h + ball.radius
        ball.pos[0] = x + w + ball.radius

        ratio = min(1.0, abs(ball_center_x - (x + w)) / ball.radius)
        transfer = ratio * ball.vel[1]
        ball.vel[0] = abs(ball.vel[0]) + abs(transfer)  # vers la droite
        ball.vel[1] = abs(ball.vel[1]) * (1 - 0.5 * ratio) # vers le bas

        # Ajustement spin selon ratio : tend vers positif si ratio grand
        # Coin droit
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=False)

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution, est_mousse, est_table)




def check_table_collision(ball, table):
    check_rect_collision(ball, table, est_mousse=False, est_table=True)



def check_ball_paddle(ball, paddle):

    center_x, center_y, vel_x, vel_y, angle, width, height = paddle.get_info()

    check_rect_collision(ball, paddle) # pour les coins supérieur et au dessus
    # pour les côtés
    x, y, w, h = paddle.get_rect()
    center_x = x + w/2

    # collision verticale avec le filet
    if y <= ball.pos[1] <= y + h and x <= ball.pos[0] <= x + w:
        # replacer la balle à gauche ou droite du filet
        if ball.pos[0] < center_x:
            ball.pos[0] = x - ball.radius
        else:
            ball.pos[0] = x + w + ball.radius

        ball.vel[0] = -ball.vel[0] * 0.9


def check_ball_net(ball, net, restitution=RESTITUTION, spin_factor=0.3, spin_damping=0.8):
    """
    Gestion de la collision avec le filet :
    - ball : instance de Ball
    - net : instance de Net
    - restitution : coefficient pour vel_x inversé
    - spin_factor : influence de l'angular_speed sur vel_y
    - spin_damping : perte de spin après collision
    """
    check_rect_collision(ball, net, est_mousse=False, est_table=True)
    x, y, w, h = net.get_rect()
    center_x = x + w/2

    # collision verticale avec le filet
    if y <= ball.pos[1] <= y + h and x <= ball.pos[0] <= x + w:
        # replacer la balle à gauche ou droite du filet
        if ball.pos[0] < center_x:
            ball.pos[0] = x - ball.radius
            direction_x = -1  # rebond vers la gauche
        else:
            ball.pos[0] = x + w + ball.radius
            direction_x = 1   # rebond vers la droite

        # transfert de vitesse selon le spin
        ratio = min(1.0, abs(ball.angular_speed) / 500.0)  # normalisation selon spin max observé, 300 c'est arbitraire, représentant le spin à partir duquel on le considère très grand

        # vel_x diminue selon le spin
        ball.vel[0] = -ball.vel[0] * restitution * (1 - 0.5 * ratio)

        # vel_y dépend du spin
        ball.vel[1] = direction_x * ball.angular_speed * spin_factor

        # réduire le spin après collision
        ball.angular_speed *= spin_damping

