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
    
    Observation (12 valeurs normalisées):
        - Position balle (x, y) normalisée [0, 1]
        - Vitesse balle (vx, vy) normalisée [-1, 1]
        - Spin balle normalisé [-1, 1]
        - Position raquette agent (x, y) normalisée [0, 1]
        - Vitesse raquette agent (vx, vy) normalisée [-1, 1]
        - Angle raquette agent normalisé [-1, 1]
        - Position adversaire (x, y) normalisée [0, 1]
    
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
        
        # Espace d'observation : 12 valeurs continues
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
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
        self.max_steps = 1000  # Timeout par épisode
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
        self.bounce_reward_given = False
        self.fault_volley = False
        self.double_hit_fault = False
        self.agent_already_hit = False
        self.service_fault = False
        self.pending_hit_reward = False
        self.ball_out_result = None
        
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
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        
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
                
                # === DETECTION BALLE OUT (Table + Marge) ===
                # Si la balle sort des limites latérales, le point est terminé immédiatement.
                if self.ball_out_result is None:
                    margin = 10
                    table_left_limit = self.table.x - margin
                    table_right_limit = self.table.x + self.table.width + margin
                    
                    # Sortie à gauche
                    if self.ball.pos[0] < table_left_limit:
                        if self.ball.bounces_left == 0:
                            # Sortie sans rebond à gauche -> Faute du tireur (celui qui a envoyé vers la gauche)
                            # Si agent est à gauche, c'est l'adversaire qui a tiré -> WIN
                            if self.agent_side == 'left':
                                self.ball_out_result = 'win'
                            else:
                                self.ball_out_result = 'loss'
                        else:
                            # Rebond valide, mais balle sort -> Le receveur (gauche) a raté -> LOSS pour gauche
                            if self.agent_side == 'left':
                                self.ball_out_result = 'loss'
                            else:
                                self.ball_out_result = 'win'
                                
                    # Sortie à droite
                    elif self.ball.pos[0] > table_right_limit:
                        if self.ball.bounces_right == 0:
                            # Sortie sans rebond à droite -> Faute du tireur
                            if self.agent_side == 'left':
                                self.ball_out_result = 'loss' # Agent a tiré trop loin
                            else:
                                self.ball_out_result = 'win' # Adversaire a tiré trop loin
                        else:
                            # Rebond valide, balle sort -> Receveur (droite) a raté -> LOSS pour droite
                            if self.agent_side == 'left':
                                self.ball_out_result = 'win'
                            else:
                                self.ball_out_result = 'loss'
                
                # Si le point est terminé par un OUT, on arrête la physique/collisions
                if self.ball_out_result is not None:
                    continue

                # Détecter le changement de côté de la balle
                net_center = WIDTH // 2
                current_side = 'left' if self.ball.pos[0] < net_center else 'right'
                
                # Si la balle change de côté
                if self.ball_side is not None and current_side != self.ball_side:
                    # Vérifier service invalide (balle traverse le filet sans avoir rebondi chez le serveur)
                    if self.ball.is_service:
                        # Qui a servi ? (celui qui a touché la balle en dernier)
                        server = self.ball.last_hit_by
                        
                        # Est-ce une faute ? (n'a pas rebondi du côté du serveur avant de passer)
                        is_fault = False
                        if server == 'left' and self.ball_side == 'left' and self.ball.bounces_left == 0:
                            is_fault = True
                        elif server == 'right' and self.ball_side == 'right' and self.ball.bounces_right == 0:
                            is_fault = True
                            
                        if is_fault:
                            # Si c'est l'agent qui a fait la faute
                            if (server == 'left' and self.agent_side == 'left') or \
                               (server == 'right' and self.agent_side == 'right'):
                                self.service_fault = True
                            # Sinon c'est l'adversaire (on pourrait lui donner une pénalité ou donner le point à l'agent)
                        else:
                            # Service réussi, le jeu continue normalement
                            self.ball.is_service = False
                    
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
                old_can_hit_agent = self.agent_paddle.can_hit
                ball_hit_agent = self._check_paddle_collision(self.agent_paddle, "agent")
                
                # Si on touche la balle
                if ball_hit_agent:
                    # Reset du flag de récompense de rebond pour le nouveau coup
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
                old_can_hit_opp = self.opponent_paddle.can_hit
                ball_hit_opponent = self._check_paddle_collision(self.opponent_paddle, "opponent")
                if ball_hit_opponent:
                    self.ball.last_hit_by = 'right' if self.agent_side == 'left' else 'left'
        
        # === Calculer la récompense ===
        reward, terminated = self._compute_reward()
        
        # === Vérifier timeout ===
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = {"steps": self.steps, "agent_hits": self.agent_hits}
        
        # Rendu si demandé
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
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
            return reward, terminated

        # Faute de volée (Obstruction)
        if self.fault_volley:
            reward = -10.0  # Faute directe
            terminated = True
            self.ball_in_play = False
            return reward, terminated
            
        # Balle sortie (détectée dans step)
        if self.ball_out_result == 'win':
            reward = 20.0   # Victoire !
            terminated = True
            self.ball_in_play = False
            return reward, terminated
        elif self.ball_out_result == 'loss':
            reward = -15.0  # Défaite (faute ou raté)
            terminated = True
            self.ball_in_play = False
            return reward, terminated
            
        # Faute de service (pas de rebond sur son côté)
        if self.service_fault:
            reward = -10.0
            terminated = True
            self.ball_in_play = False
            return reward, terminated

        # Double rebond = faute
        if self.ball.bounces_left >= 2:
            if agent_is_left:
                reward = -15.0  # Agent perd - n'a pas réussi à toucher
            else:
                reward = 20.0   # Agent gagne - adversaire a raté
            terminated = True
            self.ball_in_play = False
            return reward, terminated
        
        if self.ball.bounces_right >= 2:
            if agent_is_left:
                reward = 20.0   # Agent gagne
            else:
                reward = -15.0  # Agent perd
            terminated = True
            self.ball_in_play = False
            return reward, terminated
        
        # Balle sortie par le bas (sous la table)
        if ball_y > HEIGHT:
            ball_last_hit_by_agent = (
                (self.ball.last_hit_by == 'left' and agent_is_left) or
                (self.ball.last_hit_by == 'right' and not agent_is_left)
            )
            if ball_last_hit_by_agent:
                reward = -10.0  # Faute de l'agent (balle dans le filet ou hors table)
            else:
                reward = 15.0   # Faute de l'adversaire
            terminated = True
            self.ball_in_play = False
            return reward, terminated
        
        # === RÉCOMPENSES INTERMÉDIAIRES (jalons) ===
        
        # Déterminer la situation
        ball_on_agent_side = (ball_x < net_center) if agent_is_left else (ball_x >= net_center)
        ball_coming_to_agent = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
        ball_going_to_opponent = not ball_coming_to_agent
        
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
        
        ideal_y = TABLE_Y  # Au niveau de la table
        
        dist_to_ideal_x = abs(paddle_center_x - ideal_x)
        dist_to_ideal_y = abs(paddle_center_y - ideal_y)
        
        # Petite récompense pour être en bonne position quand la balle est loin
        if not ball_on_agent_side and not ball_coming_to_agent:
            position_score = 0.0
            if dist_to_ideal_x < 100:
                position_score += 0.002
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
            reward += 10.0  # Bonne récompense pour avoir touché
            self.pending_hit_reward = False
        
        # === JALON 5 : Balle qui rebondit chez l'adversaire ===
        # Récompenser quand la balle rebondit sur la table adverse après notre frappe
        if self.agent_hits > 0 and ball_going_to_opponent:
            if agent_is_left and self.ball.bounces_right > 0:
                if not self.bounce_reward_given:
                    reward += 50.0  # Super ! La balle est en jeu chez l'adversaire
                    self.bounce_reward_given = True
            elif not agent_is_left and self.ball.bounces_left > 0:
                if not self.bounce_reward_given:
                    reward += 50.0
                    self.bounce_reward_given = True
        
        # === PÉNALITÉS ===
        
        # Pénalité si la balle est de notre côté et qu'on est très loin
        if ball_on_agent_side and ball_is_playable and distance > ZONE_MOYENNE:
            reward -= 0.01  # Pénalité pour être trop loin

        # Récompense pour rester à hauteur de table quand la balle n'est pas jouable
        if not ball_is_playable:
            if abs(paddle_center_y - TABLE_Y) < 50:
                reward += 0.005  # Bien positionné en attente
        
        # Pénalité pour mouvements erratiques (stabilité)
        paddle_speed = np.sqrt(self.agent_paddle.vel[0]**2 + self.agent_paddle.vel[1]**2)
        if not ball_on_agent_side and not ball_coming_to_agent:
            if paddle_speed > 300:
                reward -= 0.001
        
        return reward, terminated
    
    def _get_observation(self):
        """
        Retourne l'observation normalisée.
        """
        obs = np.zeros(12, dtype=np.float32)
        
        if self.ball_in_play and self.ball is not None:
            # Position balle normalisée [0, 1] -> [-1, 1]
            obs[0] = (self.ball.pos[0] / WIDTH) * 2 - 1
            obs[1] = (self.ball.pos[1] / HEIGHT) * 2 - 1
            
            # Vitesse balle normalisée (max ~1000 px/s)
            max_vel = 1000.0
            obs[2] = np.clip(self.ball.vel[0] / max_vel, -1, 1)
            obs[3] = np.clip(self.ball.vel[1] / max_vel, -1, 1)
            
            # Spin normalisé (max ~500)
            max_spin = 500.0
            obs[4] = np.clip(self.ball.angular_speed / max_spin, -1, 1)
        
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
