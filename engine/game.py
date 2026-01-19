"""
Wrapper d'affichage pour le jeu Ping-Pong.
Gère l'interface visuelle et les inputs clavier.
La logique du jeu est entièrement gérée par PingPongEnv (environment.py).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
from config import WIDTH, HEIGHT, FPS
from graphics.renderer import draw_background, draw_table, draw_ball, draw_paddle, draw_net
from ai.environment import PingPongEnv


class Game:
    """
    Wrapper d'affichage pour le Ping-Pong.
    Utilise PingPongEnv pour la logique du jeu.
    """
    
    def __init__(self, player1_type="human", player2_type="human"):
        """
        Initialise le jeu.
        
        Args:
            player1_type: "human" ou "ai"
            player2_type: "human" ou "ai"
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ping-Pong 2D")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Environnement du jeu (gère toute la logique)
        self.env = PingPongEnv(render_mode=None)
        
        # Types de joueurs
        self.player1_type = player1_type  # "human" ou "ai"
        self.player2_type = player2_type
        
        # Affichage
        self.font = pygame.font.Font(None, 36)
        self.score_font = pygame.font.Font(None, 72)
        
        # Scores
        self.score_left = 0
        self.score_right = 0
        self.point_message = ""
        self.message_timer = 0
        
        # Pour le debug
        self.debug_timer = 0
        self.last_ball_vel = (0, 0)
        self.last_paddle_vel = (0, 0)
        self.last_spin = 0
        
        # Réinitialiser l'environnement
        self.env.reset()

    
    def run(self):
        """Boucle principale du jeu."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_events(self):
        """Gestion des entrées utilisateur."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        # Récupérer les actions des joueurs
        self.action_p1 = self._get_player1_input()
        self.action_p2 = self._get_player2_input()
    
    def _get_player1_input(self):
        """Récupère l'input du joueur 1 (gauche)."""
        if self.player1_type != "human":
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        keys = pygame.key.get_pressed()
        move_x = 0.0
        move_y = 0.0
        rotate = 0.0
        
        # Mouvement vertical (Z=haut, S=bas)
        if keys[pygame.K_z]:
            move_y = -1.0
        elif keys[pygame.K_s]:
            move_y = 1.0
        
        # Mouvement horizontal (Q=gauche, D=droite)
        if keys[pygame.K_q]:
            move_x = -1.0
        elif keys[pygame.K_d]:
            move_x = 1.0
        
        # Rotation (A=gauche, E=droite)
        if keys[pygame.K_a]:
            rotate = -1.0
        elif keys[pygame.K_e]:
            rotate = 1.0
        
        return np.array([move_x, move_y, rotate], dtype=np.float32)
    
    def _get_player2_input(self):
        """Récupère l'input du joueur 2 (droite)."""
        if self.player2_type != "human":
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        keys = pygame.key.get_pressed()
        move_x = 0.0
        move_y = 0.0
        rotate = 0.0
        
        # Mouvement vertical (O=haut, L=bas)
        if keys[pygame.K_o]:
            move_y = -1.0
        elif keys[pygame.K_l]:
            move_y = 1.0
        
        # Mouvement horizontal (K=gauche, M=droite)
        if keys[pygame.K_k]:
            move_x = -1.0
        elif keys[pygame.K_m]:
            move_x = 1.0
        
        # Rotation (I=gauche, P=droite)
        if keys[pygame.K_i]:
            rotate = -1.0
        elif keys[pygame.K_p]:
            rotate = 1.0
        
        return np.array([move_x, move_y, rotate], dtype=np.float32)

    
    def update(self):
        """Met à jour l'état du jeu via environment.py."""
        # Appeler env.step() avec les actions des deux joueurs (pas de reward en mode jeu)
        obs, terminated, info = self.env.step(
            self.action_p1,
            self.action_p2
        )
        
        # Récupérer les scores depuis l'info
        self.score_left = info.get('score_left', 0)
        self.score_right = info.get('score_right', 0)
        
        # Si le point est terminé
        if terminated:
            self.point_message = info.get('point_message', '')
            self.message_timer = 120  # 2 secondes à 60 fps
            # Réinitialiser pour le prochain point
            self.env.reset()
        
        # Décrementer le timer du message
        if self.message_timer > 0:
            self.message_timer -= 1
        
        # Debug: mettre à jour les infos affichées
        self.debug_timer += 1
        if self.debug_timer >= 30:  # 30 frames = 0.5s à 60fps
            self.debug_timer = 0
            if self.env.ball_in_play and self.env.ball:
                self.last_ball_vel = (self.env.ball.vel[0], self.env.ball.vel[1])
                self.last_spin = self.env.ball.angular_speed

    
    def draw(self):
        """Dessine les éléments à l'écran."""
        draw_background(self.screen)
        draw_table(self.screen, self.env.table)
        
        # Dessiner la balle
        if self.env.ball_in_play and self.env.ball:
            draw_ball(self.screen, self.env.ball)
        
        # Dessiner les raquettes
        draw_paddle(self.screen, self.env.agent_paddle, (255, 0, 0))
        draw_paddle(self.screen, self.env.opponent_paddle, (0, 0, 0))
        
        # Dessiner le filet
        draw_net(self.screen, self.env.net)
        
        # Afficher le score
        self._draw_score()
        
        # Affichage debug des vitesses (optionnel, en bas)
        vel_text = f"Balle: vx={self.last_ball_vel[0]:.0f} vy={self.last_ball_vel[1]:.0f}"
        spin_text = f"Spin: {self.last_spin:.0f}"
        text_surface = self.font.render(vel_text, True, (255, 255, 255))
        text_surface2 = self.font.render(spin_text, True, (0, 255, 255))
        self.screen.blit(text_surface, (10, HEIGHT - 60))
        self.screen.blit(text_surface2, (10, HEIGHT - 30))
        
        pygame.display.flip()
    
    def _draw_score(self):
        """Dessine le tableau de score en haut de l'écran."""
        # Fond semi-transparent pour le score
        score_bg = pygame.Surface((400, 80), pygame.SRCALPHA)
        score_bg.fill((0, 0, 0, 150))  # Noir semi-transparent
        self.screen.blit(score_bg, (WIDTH // 2 - 200, 10))
        
        # Scores
        score_text = f"{self.score_left}  -  {self.score_right}"
        score_surface = self.score_font.render(score_text, True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(WIDTH // 2, 40))
        self.screen.blit(score_surface, score_rect)
        
        # Labels des joueurs
        left_label = self.font.render("Joueur 1", True, (255, 100, 100))
        right_label = self.font.render("Joueur 2", True, (100, 100, 100))
        self.screen.blit(left_label, (WIDTH // 2 - 180, 60))
        self.screen.blit(right_label, (WIDTH // 2 + 80, 60))
        
        # Message temporaire (ex: "Double rebond!")
        if self.message_timer > 0 and self.point_message:
            msg_surface = self.score_font.render(self.point_message, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(WIDTH // 2, 120))
            self.screen.blit(msg_surface, msg_rect)


if __name__ == "__main__":
    game = Game(player1_type="human", player2_type="human")
    game.run()

