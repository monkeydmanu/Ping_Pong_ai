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
    RACKET_WIDTH_PX, RACKET_HEIGHT_PX, TABLE_WIDTH_PX,
    ADAPTIVE_BOUNDARY_OFFSET
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
    
    def __init__(self, render_mode=None, agent_side="left", player1_mouse_control=False):
        """
        Initialise l'environnement Ping-Pong.
        
        Args:
            render_mode (str): Mode de rendu ("human" ou "rgb_array")
            agent_side (str): Côté de l'agent ("left" ou "right")
            player1_mouse_control (bool): Si True, le joueur 1 est contrôlé à la souris
                (dans ce cas, on n'applique que la rotation, pas le mouvement)
        """
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
        
        # === Contrôle du joueur 1 ===
        self.player1_mouse_control = player1_mouse_control
        
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
        self.is_agent_service = True
        
        # Scores
        self.score_left = 0
        self.score_right = 0
        self.last_point_message = ""
        
        # Flags pour les récompenses (éviter les doublons)
        self.bounce_reward_given = False
        self.fault_volley = False
        self.double_hit_fault = False
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
            self.agent_paddle = Paddle(265, HEIGHT // 2 - 30, x_min=0, x_max=net_center) # 50
            self.opponent_paddle = Paddle(WIDTH - 60, HEIGHT // 2 - 30, x_min=net_center, x_max=WIDTH)
        else:
            self.agent_paddle = Paddle(WIDTH - 60, HEIGHT // 2 - 30, x_min=net_center, x_max=WIDTH)
            self.opponent_paddle = Paddle(50, HEIGHT // 2 - 30, x_min=0, x_max=net_center)
        
        # Randomiser le service (50% chance que l'agent serve)
        # Pour l'entraînement, on force l'agent à servir tout le temps
        self.is_agent_service = True
        # is_agent_service = np.random.choice([True, False])
        
        if self.is_agent_service:
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
                    if self.ball.bounces_right > 0:
                        has_valid_bounce = True
                    elif self.ball.bounces_left > 0:
                        has_valid_bounce = True
                    
                    # === CAS 1 : PAS DE REBOND ADVERSE (détection précoce avec marges) ===
                    if not has_valid_bounce and hitter is not None:
                        margin = 12
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
                    
                    # Déterminer de quel côté la balle tombe (avec offset adaptatif)
                    # Si la balle va à droite: offset +ADAPTIVE_BOUNDARY_OFFSET, si elle va à gauche: offset -ADAPTIVE_BOUNDARY_OFFSET
                    velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
                    net_center = WIDTH // 2 + velocity_offset
                    ball_falls_on_left = self.ball.pos[0] < net_center
                    
                    # Déterminer qui gagne selon où tombe la balle et qui a frappé en dernier
                    if last == 'left':
                        # Le joueur gauche a frappé en dernier
                        if ball_falls_on_left:
                            # Tombe de son côté → il perd
                            winner_side = 'right'
                        else:
                            # Tombe du côté adverse → il gagne
                            winner_side = 'left'
                    elif last == 'right':
                        # Le joueur droite a frappé en dernier
                        if ball_falls_on_left:
                            # Tombe du côté adverse → il gagne
                            winner_side = 'right'
                        else:
                            # Tombe de son côté → il perd
                            winner_side = 'left'
                    
                    self.point_winner_side = winner_side
                    
                    # Traduire en résultat pour l'agent
                    agent_wins = (
                        (winner_side == 'left' and agent_is_left) or
                        (winner_side == 'right' and not agent_is_left)
                    )
                    self.ball_out_result = 'win' if agent_wins else 'loss'
                
                # Si le point est terminé par un OUT, on arrête la physique/collisions
                if self.ball_out_result is not None:
                    continue

                # Collisions (table/net) avant de vérifier le changement de côté, 
                # pour que les rebonds soient comptés avant le contrôle de service.
                check_table_collision(self.ball, self.table)
                check_ball_net(self.ball, self.net)

                # Détecter le changement de côté de la balle (avec offset adaptatif basé sur la vélocité)
                # Si la balle va à droite: offset +ADAPTIVE_BOUNDARY_OFFSET, si elle va à gauche: offset -ADAPTIVE_BOUNDARY_OFFSET
                velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
                net_center = WIDTH // 2 + velocity_offset
                if self.ball is not None:
                    current_side = 'left' if self.ball.pos[0] < net_center else 'right'
                else:
                    current_side = 'left'  # par défaut
                
                # Si la balle change de côté
                if self.ball_side is not None and current_side != self.ball_side:
                    # Reset les compteurs de rebonds et can_hit
                    self.agent_paddle.can_hit = True
                    self.opponent_paddle.can_hit = True

                    if self.is_agent_service:
                        server_side = self.ball.service
                        if server_side == 'left' and current_side == 'right':
                            if self.ball.bounces_left != 1:
                                self.service_fault = True
                        elif server_side == 'right' and current_side == 'left':
                            if self.ball.bounces_right != 1:
                                self.service_fault = True
                
                        
                    if self.ball_side == 'left':
                        self.ball.bounces_left = 0
                    else:
                        self.ball.bounces_right = 0
                
                self.ball_side = current_side
                
                # === COLLISIONS AVEC RAQUETTES ===
                
                # Collision avec raquette agent
                ball_hit_agent = self._check_paddle_collision(self.agent_paddle, "agent")
                if ball_hit_agent:
                    self.bounce_reward_given = False
                    agent_paddle_side = 'left' if self.agent_side == 'left' else 'right'
                    
                    # Sauvegarder qui a frappé avant (pour détecter volley)
                    if hasattr(self.ball, 'last_hit_by'):
                        self.ball.previous_hit_by = self.ball.last_hit_by
                    
                    
                    self.pending_hit_reward = True
                    self.agent_hits += 1
                    self.ball.last_hit_by = agent_paddle_side
                
                # Collision avec raquette adversaire
                ball_hit_opponent = self._check_paddle_collision(self.opponent_paddle, "opponent")
                if ball_hit_opponent:
                    opponent_paddle_side = 'right' if self.agent_side == 'left' else 'left'
                    
                    # Sauvegarder qui a frappé avant
                    if hasattr(self.ball, 'last_hit_by'):
                        self.ball.previous_hit_by = self.ball.last_hit_by
                    
                    self.ball.last_hit_by = opponent_paddle_side
        
        # === DÉTECTION DES FAUTES POUR LES DEUX CÔTÉS ===
        terminated = False
        faults = {
            'volley_left': False,
            'volley_right': False,
            'double_bounce_left': False,
            'double_bounce_right': False,
            'out': False,
            'service_fault': False,
            'ball_out_bottom': False,
        }
        
        # Volley (obstruction) - vérifier pour les deux côtés
        if self.ball.last_hit_by in ('left', 'right'):
            # Vérifier si le joueur gauche a fait une volée
            if self.ball.last_hit_by == 'left' and self.ball.bounces_left == 0 and self.ball_side == 'left':
                # Vérifier que la balle avait été frappée par l'adversaire avant
                prev_hit = getattr(self.ball, 'previous_hit_by', None)
                if prev_hit == 'right':
                    faults['volley_left'] = True
                    self.point_winner_side = 'right'
                    terminated = True
                    self.ball_in_play = False
                    self._update_scores("Obstruction gauche!")
            
            # Vérifier si le joueur droit a fait une volée
            if self.ball.last_hit_by == 'right' and self.ball.bounces_right == 0 and self.ball_side == 'right':
                prev_hit = getattr(self.ball, 'previous_hit_by', None)
                if prev_hit == 'left':
                    faults['volley_right'] = True
                    self.point_winner_side = 'left'
                    terminated = True
                    self.ball_in_play = False
                    self._update_scores("Obstruction droite!")
        
        # Double rebond - vérifier pour les deux côtés
        if not terminated and self.ball.bounces_left >= 2:
            faults['double_bounce_left'] = True
            self.point_winner_side = 'right'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Double rebond gauche!")
        
        if not terminated and self.ball.bounces_right >= 2:
            faults['double_bounce_right'] = True
            self.point_winner_side = 'left'
            terminated = True
            self.ball_in_play = False
            self._update_scores("Double rebond droite!")
        
        # Balle sortie (latérale)
        if not terminated and self.ball_out_result is not None:
            faults['out'] = True
            # point_winner_side déjà défini dans la détection OUT
            terminated = True
            print("OUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUTTTTTTTTTTTTTTTTTTTTTTTT")
            self.ball_in_play = False
            self._update_scores("Out!")
        
        # Faute de service
        if not terminated and self.service_fault:
            faults['service_fault'] = True
            # point_winner_side à déterminer selon le serveur
            server_side = self.ball.service
            if server_side in ('left', 'right'):
                self.point_winner_side = 'right' if server_side == 'left' else 'left'
            terminated = True
            print("SERRRRRRVVVVVVVVVVVIIIIIIIIIIIIIIICEEEEEEE")
            self.ball_in_play = False
            self._update_scores("Service invalide!")
        
        
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "agent_hits": self.agent_hits,
            "winner_side": self.point_winner_side,  # 'left', 'right' ou None
            "faults": faults,
            "score_left": self.score_left,
            "score_right": self.score_right,
            "point_message": self.last_point_message,
            "ball_on_agent_side": self._is_ball_on_agent_side(),
            "ball_bounces_agent": self._get_agent_side_bounces(),
            "ball_bounces_opponent": self._get_opponent_side_bounces(),
        }
        
        # Rendu si demandé
        if self.render_mode == "human":
            self.render()
        
        return observation, terminated, info
    
    def _apply_action(self, paddle, action):
        """
        Applique une action continue à une raquette.
        
        Si player1_mouse_control est True et c'est la raquette agent (joueur 1),
        on n'applique que la rotation (move_x et move_y sont gérés par la souris).
        """
        move_x, move_y, rotate = action
        
        # Si c'est l'agent et que le contrôle souris est activé, sauter les mouvements
        # (ils sont gérés par la souris dans Game)
        if paddle == self.agent_paddle and self.player1_mouse_control:
            # Ne pas appliquer move_x et move_y, juste la rotation
            pass
        else:
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
        
        # Rotation (toujours appliquée, même en mode souris)
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
        old_pos = self.ball.pos.copy()
        check_ball_paddle(self.ball, paddle, None)
        
        # Si la position a changé, c'est qu'il y a eu collision
        if not np.array_equal(old_pos, self.ball.pos):
            self.last_hit_by = who
            return True
        return False
    
    def compute_reward(self, info):
        """
        Calcule la récompense basée sur les informations du step.
        Fonction pure appelée après step() uniquement pour l'entraînement.
        
        Args:
            info: Dictionnaire retourné par step() contenant:
                - winner_side: 'left', 'right' ou None
                - faults: dict de booléens pour chaque type de faute
                - ball_on_agent_side: bool
                - ball_bounces_agent: int
                - ball_bounces_opponent: int
                - agent_hits: int
        
        Returns:
            reward: float
        """
        reward = 0.0
        
        agent_is_left = (self.agent_side == "left")
        winner_side = info.get('winner_side')
        faults = info.get('faults', {})
        
        if self.agent_hits == 0: # priorise la touche de balle

            # === RÉCOMPENSES TERMINALES ===
            if winner_side is not None:
                # Déterminer si l'agent a gagné ou perdu
                agent_wins = (
                    (winner_side == 'left' and agent_is_left) or
                    (winner_side == 'right' and not agent_is_left)
                )
                
                # Récompenses selon le type de faute
                if faults.get('volley_left') or faults.get('volley_right'):
                    # Faute de volée
                    reward = 20.0 if agent_wins else -5.0
                    if self.render_mode == "human":
                        print(f"    {'🟢' if agent_wins else '🔴'} FIN: Volée (reward={reward})")
                
                elif faults.get('double_bounce_left') or faults.get('double_bounce_right'):
                    # Double rebond - adversaire n'a pas réussi à toucher
                    reward = 20.0 if agent_wins else -5.0
                    if self.render_mode == "human":
                        print(f"    {'🟢' if agent_wins else '🔴'} FIN: Double rebond (reward={reward})")
                
                elif faults.get('out'):
                    # Balle sortie
                    reward = 20.0 if agent_wins else -5.0
                    if self.render_mode == "human":
                        print(f"    {'🟢' if agent_wins else '🔴'} FIN: Out (reward={reward})")
                
                elif faults.get('service_fault'):
                    # Faute de service
                    reward = 20.0 if agent_wins else -5.0
                    if self.render_mode == "human":
                        print(f"    {'🟢' if agent_wins else '🔴'} FIN: Service invalide (reward={reward})")
                
                elif faults.get('double_hit'):
                    # Double touche
                    reward = 20.0 if agent_wins else -5.0
                    if self.render_mode == "human":
                        print(f"    {'🟢' if agent_wins else '🔴'} FIN: Double touche (reward={reward})")
                
                else:
                    # Cas par défaut (ne devrait pas arriver)
                    reward = 20.0 if agent_wins else -5.0
                
                return reward
        
        # === RÉCOMPENSES INTERMÉDIAIRES (jalons) ===
        # Uniquement si le point n'est pas terminé
        if winner_side is None and self.ball_in_play:
            ball_x = self.ball.pos[0]
            ball_y = self.ball.pos[1]
            net_center = WIDTH // 2
            
            # Position de la raquette agent
            paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
            paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
            
            # Déterminer la situation
            ball_on_agent_side = info.get('ball_on_agent_side', False)
            ball_coming_to_agent = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
            
            # Distance balle-raquette
            distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
            
            # Zones de proximité
            ZONE_TRES_PROCHE = 100   # pixels    # 50
            ZONE_PROCHE = 200       # pixels    # 150
            ZONE_MOYENNE = 300     # pixels    # 300

            
            ball_is_playable = ball_y > (TABLE_Y - 50)
            
            # === Proximité avec la balle ===
            if ball_on_agent_side and ball_is_playable:
                if distance < ZONE_TRES_PROCHE:
                    reward += 1 # 0.05
                elif distance < ZONE_PROCHE:
                    reward += 0.5  # 0.03
                elif distance < ZONE_MOYENNE:
                    reward += 0.1 # 0.01
                else:
                    if ball_on_agent_side:
                        reward -= 0.01
            
            # === Toucher la balle ===
            if self.pending_hit_reward: # la récompense doit être supérieur aux fautes car on fait souvent des -15 pour la faute donc +30 permet de finir avec +15
                reward += 80.0
                self.pending_hit_reward = False
            
            # === Balle qui rebondit chez l'adversaire ===
            ball_bounces_opponent = info.get('ball_bounces_opponent', 0)
            if info.get('agent_hits', 0) > 0 and ball_bounces_opponent > 0:
                if not self.bounce_reward_given:
                    reward += 200.0
                    self.bounce_reward_given = True
            
            """
            # === PÉNALITÉS ===
            if ball_on_agent_side and ball_is_playable and distance > ZONE_MOYENNE:
                reward -= 0.01
            
            if paddle_center_y > TABLE_Y + 100:
                reward -= 0.02
            
            paddle_speed = np.sqrt(self.agent_paddle.vel[0]**2 + self.agent_paddle.vel[1]**2)
            if not ball_on_agent_side and not ball_coming_to_agent:
                if paddle_speed > 300:
                    reward -= 0.001
            """
        
        return reward
    
    def _get_distance_ball_paddle_normalized(self, paddle):
        """Calcule la distance normalisée entre la balle et une raquette."""
        if not self.ball_in_play or self.ball is None:
            return 1.0  # Distance maximale si la balle n'est pas en jeu
        
        paddle_center_x = paddle.pos[0] + paddle.width / 2
        paddle_center_y = paddle.pos[1] + paddle.height / 2
        ball_x = self.ball.pos[0]
        ball_y = self.ball.pos[1]
        
        distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
        
        # Normaliser la distance (max ~WIDTH)
        normalized_distance = np.clip(distance / WIDTH, 0.0, 1.0)
        
        return normalized_distance

    def _get_observation(self):
        """
        Retourne l'observation normalisée (18 valeurs).
        
        Structure:
        [0]   Distance raquette balle
        """
        obs = np.zeros(1, dtype=np.float32)
        
        obs[0] = self._get_distance_ball_paddle_normalized(self.agent_paddle)

        return obs

    # def _get_observation(self):
    #     """
    #     Retourne l'observation normalisée (18 valeurs).
        
    #     Structure:
    #     [0-1]   Position balle (x, y)
    #     [2-3]   Vitesse balle (vx, vy)
    #     [4]     Spin balle
    #     [5-6]   Position raquette agent (x, y)
    #     [7-8]   Vitesse raquette agent (vx, vy)
    #     [9]     Angle raquette agent
    #     [10-11] Position adversaire (x, y)
    #     [12]    Balle de notre côté ? (1 = oui, -1 = non)
    #     [13]    Balle vient vers nous ? (1 = oui, -1 = non)
    #     [14]    Rebonds sur notre côté (0, 0.5, 1)
    #     [15]    Rebonds côté adverse (0, 0.5, 1)
    #     [16]    Distance balle-raquette normalisée
    #     [17]    Est-ce un service ? (1 = oui, -1 = non)
    #     """
    #     obs = np.zeros(18, dtype=np.float32)
        
    #     # Variables utiles
    #     agent_is_left = (self.agent_side == "left")
    #     # Offset adaptatif: +15 si balle va à droite, -15 si elle va à gauche
    #     velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if (self.ball_in_play and self.ball and self.ball.vel[0] > 0) else (-ADAPTIVE_BOUNDARY_OFFSET if (self.ball_in_play and self.ball and self.ball.vel[0] < 0) else 0)
    #     net_center = WIDTH // 2 + velocity_offset
    #     paddle_center_x = self.agent_paddle.pos[0] + self.agent_paddle.width / 2
    #     paddle_center_y = self.agent_paddle.pos[1] + self.agent_paddle.height / 2
        
    #     if self.ball_in_play and self.ball is not None:
    #         ball_x = self.ball.pos[0]
    #         ball_y = self.ball.pos[1]
            
    #         # Position balle normalisée [0, 1] -> [-1, 1]
    #         obs[0] = (ball_x / WIDTH) * 2 - 1
    #         obs[1] = (ball_y / HEIGHT) * 2 - 1
            
    #         # Vitesse balle normalisée (max ~1000 px/s)
    #         max_vel = 1000.0
    #         obs[2] = np.clip(self.ball.vel[0] / max_vel, -1, 1)
    #         obs[3] = np.clip(self.ball.vel[1] / max_vel, -1, 1)
            
    #         # Spin normalisé (max ~500)
    #         max_spin = 500.0
    #         obs[4] = np.clip(self.ball.angular_speed / max_spin, -1, 1)
            
    #         # === NOUVELLES VARIABLES ===
            
    #         # Balle de notre côté ? (avec offset adaptatif basé sur la vélocité)
    #         velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
    #         net_center_offset = WIDTH // 2 + velocity_offset
    #         ball_on_agent_side = (ball_x < net_center_offset) if agent_is_left else (ball_x >= net_center_offset)
    #         obs[12] = 1.0 if ball_on_agent_side else -1.0
            
    #         # Balle vient vers nous ?
    #         ball_coming = (self.ball.vel[0] < 0) if agent_is_left else (self.ball.vel[0] > 0)
    #         obs[13] = 1.0 if ball_coming else -1.0
            
    #         # Rebonds sur notre côté (0, 0.5 pour 1, 1.0 pour 2+)
    #         our_bounces = self.ball.bounces_left if agent_is_left else self.ball.bounces_right
    #         obs[14] = min(our_bounces * 0.5, 1.0)
            
    #         # Rebonds côté adverse
    #         their_bounces = self.ball.bounces_right if agent_is_left else self.ball.bounces_left
    #         obs[15] = min(their_bounces * 0.5, 1.0)
            
    #         # Distance balle-raquette normalisée (max ~WIDTH)
    #         distance = np.sqrt((paddle_center_x - ball_x)**2 + (paddle_center_y - ball_y)**2)
    #         obs[16] = np.clip(distance / WIDTH, 0, 1) * 2 - 1  # [-1, 1]
            
    #         # Est-ce un service ?
    #         obs[17] = 1.0 if self.ball.service is not None else -1.0
        
    #     # Position raquette agent normalisée
    #     obs[5] = (self.agent_paddle.pos[0] / WIDTH) * 2 - 1
    #     obs[6] = (self.agent_paddle.pos[1] / HEIGHT) * 2 - 1
        
    #     # Vitesse raquette agent normalisée
    #     max_paddle_vel = 500.0
    #     obs[7] = np.clip(self.agent_paddle.vel[0] / max_paddle_vel, -1, 1)
    #     obs[8] = np.clip(self.agent_paddle.vel[1] / max_paddle_vel, -1, 1)
        
    #     # Angle raquette normalisé [-180, 180] -> [-1, 1]
    #     obs[9] = self.agent_paddle.angle / 180.0
        
    #     # Position adversaire normalisée
    #     obs[10] = (self.opponent_paddle.pos[0] / WIDTH) * 2 - 1
    #     obs[11] = (self.opponent_paddle.pos[1] / HEIGHT) * 2 - 1
        
    #     return obs

    def _get_winner_flag(self):
        """Retourne 'agent', 'opponent' ou None selon le vainqueur courant."""
        if self.point_winner_side is None:
            return None
        if (self.point_winner_side == 'left' and self.agent_side == 'left') or \
           (self.point_winner_side == 'right' and self.agent_side == 'right'):
            return "agent"
        return "opponent"
    
    def _is_ball_on_agent_side(self):
        """Retourne True si la balle est du côté de l'agent (avec offset adaptatif)."""
        if not self.ball_in_play or self.ball is None:
            return False
        # Offset adaptatif: +ADAPTIVE_BOUNDARY_OFFSET si balle va à droite, -ADAPTIVE_BOUNDARY_OFFSET si elle va à gauche
        velocity_offset = ADAPTIVE_BOUNDARY_OFFSET if self.ball.vel[0] > 0 else -ADAPTIVE_BOUNDARY_OFFSET
        net_center = WIDTH // 2 + velocity_offset
        agent_is_left = (self.agent_side == "left")
        ball_x = self.ball.pos[0]
        return (ball_x < net_center) if agent_is_left else (ball_x >= net_center)
    
    def _get_agent_side_bounces(self):
        """Retourne le nombre de rebonds du côté de l'agent."""
        if not self.ball_in_play or self.ball is None:
            return 0
        agent_is_left = (self.agent_side == "left")
        return self.ball.bounces_left if agent_is_left else self.ball.bounces_right
    
    def _get_opponent_side_bounces(self):
        """Retourne le nombre de rebonds du côté de l'adversaire."""
        if not self.ball_in_play or self.ball is None:
            return 0
        agent_is_left = (self.agent_side == "left")
        return self.ball.bounces_right if agent_is_left else self.ball.bounces_left
    
    def _update_scores(self, message=""):
        """Mise à jour des scores quand un point est marqué."""
        if self.point_winner_side == 'left':
            self.score_left += 1
        elif self.point_winner_side == 'right':
            self.score_right += 1
        
        self.last_point_message = message
    
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
