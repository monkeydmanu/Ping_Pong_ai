"""
Environnement Gymnasium pour le Ping-Pong.
Compatible avec Stable-Baselines3 pour l'entraînement PPO.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import (
    WIDTH, HEIGHT, FPS, TABLE_Y, PIXELS_PER_METER,
    RACKET_WIDTH_PX, RACKET_HEIGHT_PX, TABLE_WIDTH_PX
)
from core.ball import Ball, spawn_ball_left, spawn_ball_right
from core.paddle import Paddle
from core.net import Net
from core.table import Table
from engine.collision import check_ball_paddle, check_ball_net, check_table_collision


class PingPongEnv(gym.Env):
    """
    Environnement Ping-Pong pour reinforcement learning.
    
    Observation (18 valeurs normalisées [-1, 1]):
        [0-1]   Position balle (x, y)
        [2-3]   Vitesse balle (vx, vy)
        [4]     Spin balle
        [5-6]   Position raquette agent (x, y)
        [7-8]   Vitesse raquette agent (vx, vy)
        [9]     Angle raquette agent
        [10-11] Position adversaire (x, y)
        [12]    Balle de notre côté ? (1=oui, -1=non)
        [13]    Balle vient vers nous ? (1=oui, -1=non)
        [14]    Rebonds sur notre côté (0, 0.5, 1)
        [15]    Rebonds côté adverse (0, 0.5, 1)
        [16]    Distance balle-raquette normalisée
        [17]    Est-ce un service ? (1=oui, -1=non)
    
    Actions (3 valeurs continues [-1, 1]):
        - move_x : mouvement horizontal
        - move_y : mouvement vertical
        - rotate : rotation de la raquette
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, agent_side="left"):
        super().__init__()
        
        self.render_mode = render_mode
        self.agent_side = agent_side  # "left" ou "right"
        
        # Espace d'observation : 18 valeurs continues
        # [0-1]   Position balle (x, y)
        # [2-3]   Vitesse balle (vx, vy)
        # [4]     Spin balle
        # [5-6]   Position raquette agent (x, y)
        # [7-8]   Vitesse raquette agent (vx, vy)
        # [9]     Angle raquette agent
        # [10-11] Position adversaire (x, y)
        # [12]    Balle de notre côté ? (1 = oui, -1 = non)
        # [13]    Balle vient vers nous ? (1 = oui, -1 = non)
        # [14]    Rebonds sur notre côté (0, 0.5, 1)
        # [15]    Rebonds côté adverse (0, 0.5, 1)
        # [16]    Distance balle-raquette normalisée
        # [17]    Est-ce un service ? (1 = oui, -1 = non)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(18,),
            dtype=np.float32
        )
        
        # Espace d'action : 3 valeurs continues [-1, 1]
        # [move_x, move_y, rotate]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Initialisation Pygame (optionnel pour le rendu)
        self.screen = None
        self.clock = None
        
        # Objets du jeu
        self.table = None
        self.net = None
        self.ball = None
        self.agent_paddle = None
        self.opponent_paddle = None
        
        # État du jeu
        self.steps = 0
        self.last_hit_by = None  # "agent" ou "opponent"
        self.ball_in_play = False
        self.ball_side = None  # 'left' ou 'right' - côté actuel de la balle
        
        # Flags pour les récompenses (éviter les doublons)
        self.bounce_reward_given = False
        self.fault_volley = False
        self.double_hit_fault = False
        self.agent_already_hit = False
        self.service_fault = False
        self.pending_hit_reward = False
        self.ball_out_result = None  # 'win' ou 'loss'
        self.point_winner_side = None  # 'left' ou 'right'
        
        
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode."""
        super().reset(seed=seed)
        
        # Créer les objets du jeu
        self.table = Table()
        self.net = Net()
        net_center = WIDTH // 2
        
        # Créer les raquettes selon le côté de l'agent
        if self.agent_side == "left":
            self.agent_paddle = Paddle(50, HEIGHT // 2 - 30, x_min=0, x_max=net_center)
            self.opponent_paddle = Paddle(WIDTH - 60, HEIGHT // 2 - 30, x_min=net_center, x_max=WIDTH)
        else:
            self.agent_paddle = Paddle(WIDTH - 60, HEIGHT // 2 - 30, x_min=net_center, x_max=WIDTH)
            self.opponent_paddle = Paddle(50, HEIGHT // 2 - 30, x_min=0, x_max=net_center)
        
        # Randomiser le service (50% chance que l'agent serve)
        # Pour l'entraînement, on force l'agent à servir tout le temps
        is_agent_service = True
        # is_agent_service = np.random.choice([True, False])
        
        if is_agent_service:
            # L'agent sert
            if self.agent_side == "left":
                self.ball = spawn_ball_left(self.table)
            else:
                self.ball = spawn_ball_right(self.table)
        else:
            # L'adversaire sert
            if self.agent_side == "left":
                self.ball = spawn_ball_right(self.table)
            else:
                self.ball = spawn_ball_left(self.table)
        
        self.ball_in_play = True
        self.steps = 0
        self.last_hit_by = None
        self.agent_hits = 0  # Compteur de frappes de l'agent
        
        # Reset des flags de récompenses
        self.bounce_reward_given = False # flag temporaire qui s'active et se désactive la balle touche la table adverse, pour donner une récompense une seule fois
        self.fault_volley = False # touche la balle en volée
        self.double_hit_fault = False
        self.agent_already_hit = False # savoir si l'agent a déjà touché la balle sans qu'elle change de côté
        self.service_fault = False
        self.pending_hit_reward = False
        self.ball_out_result = None # flag temporaire qui s'active et se désactive quand on touche une balle, pour donner une récompense une seule fois
        self.point_winner_side = None
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action, opponent_action=None):
        """
        Exécute une action et retourne le nouvel état.
        
        Args:
            action: Action de l'agent principal
            opponent_action: (Optionnel) Action de l'adversaire. 
                             Si None, utilise l'IA interne _get_opponent_action().
        
        Returns:
            observation, reward, terminated, info
        """
        self.steps += 1
        self.point_winner_side = None  # reset du vainqueur pour ce step
        agent_is_left = (self.agent_side == "left")
        
        # === Appliquer l'action de l'agent ===
        self._apply_action(self.agent_paddle, action)
        
        # === Appliquer l'action de l'adversaire ===
        if opponent_action is None:
            # IA simple par défaut
            actual_opponent_action = self._get_opponent_action()
        else:
            # Action fournie (pour le self-play ou 2ème agent)
            actual_opponent_action = opponent_action
            
        self._apply_action(self.opponent_paddle, actual_opponent_action)
        
        # === Mettre à jour la physique ===
        dt = 1.0 / FPS
        self.agent_paddle.update(dt)
        self.opponent_paddle.update(dt)
        
        if self.ball_in_play:
            # Sub-stepping pour éviter le tunneling (balle qui traverse la raquette)
            n_substeps = 4
            dt_sub = dt / n_substeps
            
            for _ in range(n_substeps):
                self.ball.update(dt=dt_sub)
                
                # === DETECTION BALLE OUT ===
                # Deux cas distincts :
                # 1. Pas de rebond adverse : balle sort des limites + marge -> point fini
                # 2. Avec rebond adverse : balle sort des limites de l'écran -> point fini
                if self.ball_out_result is None:
                    hitter = self.ball.last_hit_by
                    
                    # Cas où on ne sait pas qui a frappé (au début)
                    if hitter not in ('left', 'right'):
                        hitter = None
                    
                    # Vérifier s'il y a eu un rebond sur le côté adverse
                    has_valid_bounce = False
                    if hitter == 'left' and self.ball.bounces_right > 0:
                        has_valid_bounce = True
                    elif hitter == 'right' and self.ball.bounces_left > 0:
                        has_valid_bounce = True
                    
                    # === CAS 1 : PAS DE REBOND ADVERSE (détection précoce avec marges) ===
                    if not has_valid_bounce and hitter is not None:
                        margin = 15
                        table_left_limit = self.table.x - margin
                        table_right_limit = self.table.x + self.table.width + margin
                        
                        side_out = None
                        if self.ball.pos[0] < table_left_limit:
                            side_out = 'left'
                        elif self.ball.pos[0] > table_right_limit:
                            side_out = 'right'
                        
                        if side_out is not None:
                            # Pas de rebond adverse -> faute du frappeur
                            winner_side = 'right' if hitter == 'left' else 'left'
                            self.point_winner_side = winner_side
                            
                            # Traduire en résultat pour l'agent
                            agent_wins = (
                                (winner_side == 'left' and self.agent_side == 'left') or
                                (winner_side == 'right' and self.agent_side == 'right')
                            )
                            self.ball_out_result = 'win' if agent_wins else 'loss'
                    
                    # === CAS 2 : AVEC REBOND ADVERSE (attendre sortie complète de l'écran) ===
                    elif has_valid_bounce and hitter is not None:
                        # Limites de l'écran (sans marge)
                        screen_left = 0
                        screen_right = WIDTH
                        
                        side_out = None
                        if self.ball.pos[0] < screen_left:
                            side_out = 'left'
                        elif self.ball.pos[0] > screen_right:
                            side_out = 'right'
                        
                        if side_out is not None:
                            # Il y a eu un rebond côté adverse
                            if side_out == hitter:
                                # Sortie du côté du frappeur -> faute du frappeur
                                winner_side = 'right' if hitter == 'left' else 'left'
                            else:
                                # Sortie du côté du receveur -> point au frappeur
                                winner_side = hitter

                            self.point_winner_side = winner_side
                            
                            # Traduire en résultat pour l'agent
                            agent_wins = (
                                (winner_side == 'left' and self.agent_side == 'left') or
                                (winner_side == 'right' and self.agent_side == 'right')
                            )
                            self.ball_out_result = 'win' if agent_wins else 'loss'

                # Sortie par le bas (sous la table) détectée dans step
                if self.ball_out_result is None and self.ball.pos[1] > HEIGHT:
                    last = self.ball.last_hit_by
                    ball_last_hit_by_agent = (
                        (last == 'left' and agent_is_left) or
                        (last == 'right' and not agent_is_left)
                    )

                    if ball_last_hit_by_agent:
                        winner_side = 'right' if agent_is_left else 'left'
                        self.ball_out_result = 'loss'
                    else:
                        winner_side = 'left' if agent_is_left else 'right'
                        self.ball_out_result = 'win'

                    self.point_winner_side = winner_side
                
                # Si le point est terminé par un OUT, on arrête la physique/collisions
                if self.ball_out_result is not None:
                    continue

                # Détecter le changement de côté de la balle
                net_center = WIDTH // 2
                current_side = 'left' if self.ball.pos[0] < net_center else 'right'
                
                # Si la balle change de côté
                if self.ball_side is not None and current_side != self.ball_side:
                    # Vérifier service invalide (balle traverse le filet sans avoir rebondi chez le serveur)
                    if self.ball.service is not None:
                        # Vérifier que la balle a rebondi du côté du serveur
                        server_side = self.ball.service
                        
                        # Est-ce une faute de service ? (n'a pas rebondi du côté du serveur avant de passer)
                        is_fault = False
                        if server_side == 'left' and self.ball.bounces_left == 0:
                            is_fault = True
                        elif server_side == 'right' and self.ball.bounces_right == 0:
                            is_fault = True
                            
                        if is_fault:
                            # Si c'est l'agent qui a fait la faute de service
                            if (server_side == 'left' and self.agent_side == 'left') or \
                               (server_side == 'right' and self.agent_side == 'right'):
                                self.service_fault = True
                            # Sinon c'est l'adversaire (on pourrait lui donner une pénalité ou donner le point à l'agent)
                        else:
                            # Service réussi, le jeu continue normalement
                            self.ball.service = None  # Service terminé
                    
                    # Reset les compteurs de rebonds et can_hit
                    self.agent_paddle.can_hit = True
                    self.opponent_paddle.can_hit = True
                    
                    # Reset du flag de double touche quand la balle change de côté
                    if current_side != self.agent_side:
                        self.agent_already_hit = False
                        
                    if self.ball_side == 'left':
                        self.ball.bounces_left = 0
                    else:
                        self.ball.bounces_right = 0
                
                self.ball_side = current_side
                
                # Collisions
                check_table_collision(self.ball, self.table)
                check_ball_net(self.ball, self.net)
                
                # Collision avec raquette agent
                ball_hit_agent = self._check_paddle_collision(self.agent_paddle, "agent")
                
                # Si on touche la balle
                if ball_hit_agent:
                    # Reset du flag de récompense de rebond pour le nouveau coup
                    # passera à True quand la balle touchera la table adverse
                    self.bounce_reward_given = False
                    
                    # Vérifier si on a déjà touché la balle sans qu'elle change de côté
                    if self.agent_already_hit:
                        self.double_hit_fault = True
                    else:
                        self.agent_already_hit = True
                        self.pending_hit_reward = True
                        self.agent_hits += 1
                        self.ball.last_hit_by = 'left' if self.agent_side == 'left' else 'right'
                        
                        # === DETECTION VOLLEY (OBSTRUCTION) ===
                        # Si la balle vient de l'adversaire et qu'on la touche avant qu'elle rebondisse
                        opponent_side = 'right' if self.agent_side == 'left' else 'left'
                        
                        # On vérifie seulement si c'est l'adversaire qui a frappé en dernier
                        if self.ball.last_hit_by == opponent_side:
                            is_volley = False
                            if self.agent_side == 'left':
                                if self.ball.bounces_left == 0:
                                    is_volley = True
                            else:
                                if self.ball.bounces_right == 0:
                                    is_volley = True
                            
                            if is_volley:
                                self.fault_volley = True
                
                # Collision avec raquette adversaire
                ball_hit_opponent = self._check_paddle_collision(self.opponent_paddle, "opponent")
                if ball_hit_opponent:
                    self.ball.last_hit_by = 'right' if self.agent_side == 'left' else 'left'
        
        # === Calculer la récompense ===
        reward, terminated = self._compute_reward()
        
        
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "agent_hits": self.agent_hits,
            "winner": self._get_winner_flag(),
        }
        
        # Rendu si demandé
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, info
    
    def _apply_action(self, paddle, action):
        """Applique une action continue à une raquette."""
        move_x, move_y, rotate = action
        
        # Mouvement horizontal
        if move_x > 0.3:
            paddle.move_right()
        elif move_x < -0.3:
            paddle.move_left()
        else:
            paddle.stop_horizontal()
        
        # Mouvement vertical
        if move_y > 0.3:
            paddle.move_down()
        elif move_y < -0.3:
            paddle.move_up()
        else:
            paddle.stop_vertical()
        
        # Rotation
        if rotate > 0.3:
            paddle.rotate_right(1)
        elif rotate < -0.3:
            paddle.rotate_left(1)
    
    def _get_opponent_action(self):
        """
        IA simple pour l'adversaire : suit la balle en Y.
        À remplacer par un autre agent entraîné pour le self-play.
        """
        if not self.ball_in_play:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Suivre la balle en Y
        paddle_center_y = self.opponent_paddle.pos[1] + self.opponent_paddle.height / 2
        ball_y = self.ball.pos[1]
        
        move_y = 0.0
        if ball_y < paddle_center_y - 20:
            move_y = -1.0  # Monter
        elif ball_y > paddle_center_y + 20:
            move_y = 1.0   # Descendre
        
        # Se rapprocher de la balle en X (simple)
        paddle_x = self.opponent_paddle.pos[0]
        ball_x = self.ball.pos[0]
        
        move_x = 0.0
        if self.agent_side == "left":
            # Adversaire à droite, avancer vers la balle si elle approche
            if ball_x > WIDTH // 2 and ball_x < paddle_x - 50:
                move_x = -0.5
        else:
            # Adversaire à gauche
            if ball_x < WIDTH // 2 and ball_x > paddle_x + 50:
                move_x = 0.5
        
        return np.array([move_x, move_y, 0.0], dtype=np.float32)
    
    def _check_paddle_collision(self, paddle, who):
        """Vérifie la collision balle-raquette et met à jour last_hit_by."""
        old_cooldown = self.ball.collision_cooldown
        check_ball_paddle(self.ball, paddle, None)
        
        # Si le cooldown a changé, c'est qu'il y a eu collision
        if self.ball.collision_cooldown > old_cooldown or \
           (old_cooldown == 0 and self.ball.collision_cooldown > 0):
            self.last_hit_by = who
            return True
        return False
    
    def _compute_reward(self):
        """
        Calcule la récompense et détermine si l'épisode est terminé.
        
        Système de récompenses par jalons :
        - Jalons positifs : être prêt → être proche → toucher → faire rebondir chez l'adversaire → gagner
        - Jalons négatifs : rater la balle → faire une faute → perdre le point
        
        Returns:
            (reward, terminated)
        """
        reward = 0.0
        terminated = False
        
        if not self.ball_in_play:
            return reward, terminated
        
        ball_x = self.ball.pos[0]
        ball_y = self.ball.pos[1]
        
        # Déterminer le côté de l'agent
        agent_is_left = (self.agent_side == "left")
        net_center = WIDTH // 2
        
        # Position de la raquette agent
        paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
        paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
        
        # === RÉCOMPENSES TERMINALES (fin d'épisode) ===
        
        # Faute de double touche
        if self.double_hit_fault:
            reward = -10.0  # Faute directe
            terminated = True
            self.ball_in_play = False
            # Faute de l'agent -> adversaire gagne
            if self.point_winner_side is None:
                self.point_winner_side = 'right' if agent_is_left else 'left'
            if self.render_mode == "human":
                print(f"    🔴 FIN: Double touche (reward={reward})")
            return reward, terminated

        # Faute de volée (Obstruction)
        if self.fault_volley:
            reward = -10.0  # Faute directe
            terminated = True
            self.ball_in_play = False
            if self.point_winner_side is None:
                self.point_winner_side = 'right' if agent_is_left else 'left'
            if self.render_mode == "human":
                print(f"    🔴 FIN: Volée interdite (reward={reward})")
            return reward, terminated
            
        # Balle sortie (détectée dans step)
        if self.ball_out_result == 'win':
            reward = 20.0   # Victoire !
            terminated = True
            self.ball_in_play = False
            # point_winner_side est fixé dans step; fallback au cas où
            if self.point_winner_side is None:
                self.point_winner_side = 'left' if agent_is_left else 'right'
            if self.render_mode == "human":
                print(f"    🟢 FIN: Balle out - VICTOIRE (reward={reward})")
            return reward, terminated
        elif self.ball_out_result == 'loss':
            reward = -15.0  # Défaite (faute ou raté)
            terminated = True
            self.ball_in_play = False
            if self.point_winner_side is None:
                self.point_winner_side = 'right' if agent_is_left else 'left'
            if self.render_mode == "human":
                print(f"    🔴 FIN: Balle out - DÉFAITE (reward={reward})")
            return reward, terminated
            
        # Faute de service (pas de rebond sur son côté)
        if self.service_fault:
            reward = -10.0
            terminated = True
            self.ball_in_play = False
            if self.point_winner_side is None:
                self.point_winner_side = 'right' if agent_is_left else 'left'
            if self.render_mode == "human":
                print(f"    🔴 FIN: Faute de service (reward={reward})")
            return reward, terminated

        # Double rebond = faute
        if self.ball.bounces_left >= 2:
            if agent_is_left:
                reward = -15.0  # Agent perd - n'a pas réussi à toucher
            else:
                reward = 20.0   # Agent gagne - adversaire a raté
            terminated = True
            self.ball_in_play = False
            if self.point_winner_side is None:
                self.point_winner_side = 'right'
            if self.render_mode == "human":
                print(f"    {'🔴' if reward < 0 else '🟢'} FIN: Double rebond gauche (reward={reward})")
            return reward, terminated
        
        if self.ball.bounces_right >= 2:
            if agent_is_left:
                reward = 20.0   # Agent gagne
            else:
                reward = -15.0  # Agent perd
            terminated = True
            self.ball_in_play = False
            if self.point_winner_side is None:
                self.point_winner_side = 'left'
            if self.render_mode == "human":
                print(f"    {'🔴' if reward < 0 else '🟢'} FIN: Double rebond droite (reward={reward})")
            return reward, terminated
        
        # === RÉCOMPENSES INTERMÉDIAIRES (jalons) ===
        
        # Déterminer la situation
        ball_on_agent_side = (ball_x < net_center) if agent_is_left else (ball_x >= net_center)
        ball_coming_to_agent = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
        
        # Distance balle-raquette
        distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
        
        # Zones de proximité
        ZONE_TRES_PROCHE = 50   # pixels
        ZONE_PROCHE = 150       # pixels
        ZONE_MOYENNE = 300      # pixels
        
        # === JALON 1 : Position de préparation ===
        # Récompense pour être dans une bonne position d'attente
        if agent_is_left:
            ideal_x = 80  # Proche du bord gauche
        else:
            ideal_x = WIDTH - 80  # Proche du bord droit
        
        ideal_y = TABLE_Y - 50  # Au niveau de la table
        
        dist_to_ideal_x = abs(paddle_center_x - ideal_x)
        dist_to_ideal_y = abs(paddle_center_y - ideal_y)
        
        # Petite récompense pour être en bonne position quand la balle est loin
        if not ball_on_agent_side and not ball_coming_to_agent:
            position_score = 0.0
            if dist_to_ideal_y < 80:
                position_score += 0.002
            reward += position_score
        
        # === JALON 2 : Tracking de la balle ===
        # Quand la balle vient vers l'agent ET est de son côté, récompenser le suivi en Y
        # IMPORTANT: On ne récompense PAS si la balle est encore en l'air au-dessus de la table
        ball_is_playable = ball_y > (TABLE_Y - 50)  # Balle à hauteur jouable
        
        if ball_on_agent_side and ball_is_playable:
            y_diff = abs(paddle_center_y - ball_y)
            
            if y_diff < 30:  # Très bien aligné
                reward += 0.01
            elif y_diff < 60:  # Bien aligné
                reward += 0.005
            elif y_diff > 150:  # Mal aligné - petite pénalité
                reward -= 0.002
        
        # === JALON 3 : Proximité avec la balle ===
        # Récompenser d'être proche SEULEMENT quand la balle est de son côté ET jouable
        if ball_on_agent_side and ball_is_playable:
            if distance < ZONE_TRES_PROCHE:
                reward += 0.02  # Très proche, prêt à frapper
            elif distance < ZONE_PROCHE:
                reward += 0.01  # Proche
            elif distance < ZONE_MOYENNE:
                reward += 0.003  # Distance moyenne
        
        # === JALON 4 : Toucher la balle ===
        # Grosse récompense pour avoir frappé la balle
        # On utilise agent_hits qui est incrémenté UNIQUEMENT lors d'une vraie collision
        
        # Récompense pour toucher (une seule fois par frappe)
        if self.pending_hit_reward:
            reward += 30.0  # Bonne récompense pour avoir touché
            self.pending_hit_reward = False
        
        # === JALON 5 : Balle qui rebondit chez l'adversaire ===
        # Récompenser quand la balle rebondit sur la table adverse après notre frappe
        # Note: après rebond, la balle peut revenir vers nous, donc on ne conditionne pas sur ball_going_to_opponent
        if self.agent_hits > 0:
            if agent_is_left and self.ball.bounces_right > 0:
                if not self.bounce_reward_given:
                    reward += 40.0  # Super ! La balle a rebondi chez l'adversaire
                    self.bounce_reward_given = True
            elif not agent_is_left and self.ball.bounces_left > 0:
                if not self.bounce_reward_given:
                    reward += 40.0
                    self.bounce_reward_given = True
        
        # === PÉNALITÉS ===
        
        # Pénalité si la balle est de notre côté et qu'on est très loin
        if ball_on_agent_side and ball_is_playable and distance > ZONE_MOYENNE:
            reward -= 0.01  # Pénalité pour être trop loin
        
        # Pénalité pour plonger trop bas (sous la table)
        if paddle_center_y > TABLE_Y + 100:
            reward -= 0.02  # Ne pas descendre trop bas
        
        # Pénalité pour mouvements erratiques (stabilité)
        paddle_speed = np.sqrt(self.agent_paddle.vel[0]**2 + self.agent_paddle.vel[1]**2)
        if not ball_on_agent_side and not ball_coming_to_agent:
            if paddle_speed > 300:
                reward -= 0.001
        
        return reward, terminated
    
    def _get_observation(self):
        """
        Retourne l'observation normalisée (18 valeurs).
        
        Structure:
        [0-1]   Position balle (x, y)
        [2-3]   Vitesse balle (vx, vy)
        [4]     Spin balle
        [5-6]   Position raquette agent (x, y)
        [7-8]   Vitesse raquette agent (vx, vy)
        [9]     Angle raquette agent
        [10-11] Position adversaire (x, y)
        [12]    Balle de notre côté ? (1 = oui, -1 = non)
        [13]    Balle vient vers nous ? (1 = oui, -1 = non)
        [14]    Rebonds sur notre côté (0, 0.5, 1)
        [15]    Rebonds côté adverse (0, 0.5, 1)
        [16]    Distance balle-raquette normalisée
        [17]    Est-ce un service ? (1 = oui, -1 = non)
        """
        obs = np.zeros(18, dtype=np.float32)
        
        # Variables utiles
        agent_is_left = (self.agent_side == "left")
        net_center = WIDTH // 2
        paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
        paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
        
        if self.ball_in_play and self.ball is not None:
            ball_x = self.ball.pos[0]
            ball_y = self.ball.pos[1]
            
            # Position balle normalisée [0, 1] -> [-1, 1]
            obs[0] = (ball_x / WIDTH) * 2 - 1
            obs[1] = (ball_y / HEIGHT) * 2 - 1
            
            # Vitesse balle normalisée (max ~1000 px/s)
            max_vel = 1000.0
            obs[2] = np.clip(self.ball.vel[0] / max_vel, -1, 1)
            obs[3] = np.clip(self.ball.vel[1] / max_vel, -1, 1)
            
            # Spin normalisé (max ~500)
            max_spin = 500.0
            obs[4] = np.clip(self.ball.angular_speed / max_spin, -1, 1)
            
            # === NOUVELLES VARIABLES ===
            
            # Balle de notre côté ?
            ball_on_agent_side = (ball_x < net_center) if agent_is_left else (ball_x >= net_center)
            obs[12] = 1.0 if ball_on_agent_side else -1.0
            
            # Balle vient vers nous ?
            ball_coming = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
            obs[13] = 1.0 if ball_coming else -1.0
            
            # Rebonds sur notre côté (0, 0.5 pour 1, 1.0 pour 2+)
            our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
            obs[14] = min(our_bounces * 0.5, 1.0)
            
            # Rebonds côté adverse
            their_bounces = self.ball.bounces_right if agent_is_left else self.ball.bounces_left
            obs[15] = min(their_bounces * 0.5, 1.0)
            
            # Distance balle-raquette normalisée (max ~WIDTH)
            distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
            obs[16] = np.clip(distance / WIDTH, 0, 1) * 2 - 1  # [-1, 1]
            
            # Est-ce un service ?
            obs[17] = 1.0 if self.ball.service is not None else -1.0
        
        # Position raquette agent normalisée
        obs[5] = (self.agent_paddle.pos[0] / WIDTH) * 2 - 1
        obs[6] = (self.agent_paddle.pos[1] / HEIGHT) * 2 - 1
        
        # Vitesse raquette agent normalisée
        max_paddle_vel = 500.0
        obs[7] = np.clip(self.agent_paddle.vel[0] / max_paddle_vel, -1, 1)
        obs[8] = np.clip(self.agent_paddle.vel[1] / max_paddle_vel, -1, 1)
        
        # Angle raquette normalisé [-180, 180] -> [-1, 1]
        obs[9] = self.agent_paddle.angle / 180.0
        
        # Position adversaire normalisée
        obs[10] = (self.opponent_paddle.pos[0] / WIDTH) * 2 - 1
        obs[11] = (self.opponent_paddle.pos[1] / HEIGHT) * 2 - 1
        
        return obs

    def _get_winner_flag(self):
        """Retourne 'agent', 'opponent' ou None selon le vainqueur courant."""
        if self.point_winner_side is None:
            return None
        if (self.point_winner_side == 'left' and self.agent_side == 'left') or \
           (self.point_winner_side == 'right' and self.agent_side == 'right'):
            return "agent"
        return "opponent"
    
    def render(self):
        """Affiche le jeu avec Pygame."""
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Ping-Pong RL Training")
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        
        # Import des fonctions de rendu
        from graphics.renderer import (
            draw_background, draw_table, draw_ball, 
            draw_paddle, draw_net
        )
        
        draw_background(self.screen)
        draw_table(self.screen, self.table)
        
        if self.ball_in_play and self.ball is not None:
            draw_ball(self.screen, self.ball)
        
        # Agent en rouge, adversaire en noir
        draw_paddle(self.screen, self.agent_paddle, (255, 0, 0))
        draw_paddle(self.screen, self.opponent_paddle, (0, 0, 0))
        draw_net(self.screen, self.net)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        """Ferme l'environnement."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
