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
    RACKET_WIDTH_PX, RACKET_HEIGHT_PX
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
        
        # Spawn la balle du côté de l'agent (il sert)
        if self.agent_side == "left":
            self.ball = spawn_ball_left(self.table)
        else:
            self.ball = spawn_ball_right(self.table)
        
        self.ball_in_play = True
        self.steps = 0
        self.last_hit_by = None
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Exécute une action et retourne le nouvel état.
        
        Args:
            action: np.array de shape (3,) avec [move_x, move_y, rotate]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        
        # === Appliquer l'action de l'agent ===
        self._apply_action(self.agent_paddle, action)
        
        # === Appliquer l'action de l'adversaire (IA simple pour commencer) ===
        opponent_action = self._get_opponent_action()
        self._apply_action(self.opponent_paddle, opponent_action)
        
        # === Mettre à jour la physique ===
        dt = 1.0 / FPS
        self.agent_paddle.update(dt)
        self.opponent_paddle.update(dt)
        
        if self.ball_in_play:
            self.ball.update()
            
            # Collisions
            check_table_collision(self.ball, self.table)
            check_ball_net(self.ball, self.net)
            
            # Collision avec raquette agent
            ball_hit_agent = self._check_paddle_collision(self.agent_paddle, "agent")
            # Collision avec raquette adversaire
            ball_hit_opponent = self._check_paddle_collision(self.opponent_paddle, "opponent")
        
        # === Calculer la récompense ===
        reward, terminated = self._compute_reward()
        
        # === Vérifier timeout ===
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = {"steps": self.steps}
        
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
        
        Returns:
            (reward, terminated)
        """
        reward = 0.0
        terminated = False
        
        if not self.ball_in_play:
            return reward, terminated
        
        ball_x = self.ball.pos[0]
        ball_y = self.ball.pos[1]
        
        # === Balle sortie du terrain ===
        
        # Balle sortie par la gauche
        if ball_x < 0:
            if self.agent_side == "left":
                reward = -10.0  # Agent a perdu le point
            else:
                reward = 10.0   # Agent a marqué
            terminated = True
            self.ball_in_play = False
        
        # Balle sortie par la droite
        elif ball_x > WIDTH:
            if self.agent_side == "right":
                reward = -10.0  # Agent a perdu le point
            else:
                reward = 10.0   # Agent a marqué
            terminated = True
            self.ball_in_play = False
        
        # Balle sortie par le bas (sous la table)
        elif ball_y > HEIGHT:
            # Qui a fait la faute ?
            if self.last_hit_by == "agent":
                reward = -5.0  # Faute de l'agent
            else:
                reward = 5.0   # Faute de l'adversaire
            terminated = True
            self.ball_in_play = False
        
        # === Récompenses intermédiaires ===
        else:
            # Récompense pour avoir frappé la balle
            if self.last_hit_by == "agent":
                reward += 0.1
            
            # Récompense pour se rapprocher de la balle
            paddle_center = np.array([
                self.agent_paddle.pos[0] + self.agent_paddle.width / 2,
                self.agent_paddle.pos[1] + self.agent_paddle.height / 2
            ])
            ball_pos = np.array([ball_x, ball_y])
            distance = np.linalg.norm(paddle_center - ball_pos)
            
            # Normaliser la distance
            max_distance = np.sqrt(WIDTH**2 + HEIGHT**2)
            normalized_distance = distance / max_distance
            
            # Petite récompense pour être proche de la balle
            reward += 0.01 * (1.0 - normalized_distance)
        
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
