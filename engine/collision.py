"""
Fonctions de collision balle ↔ raquette / filet / table
"""

from config import RESTITUTION, TABLE_Y

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

def reduction_speed(vx, vy, restitution):
    vx *= restitution
    vy *= restitution
    return vx, vy

def check_rect_collision(ball, rectangle, restitution=RESTITUTION, spin_factor=0.2):
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

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution)

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
        
        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution)

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

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution)

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

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution)

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

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], restitution)




def check_table_collision(ball, table):
    check_rect_collision(ball, table)



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
    check_rect_collision(ball, net)
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

