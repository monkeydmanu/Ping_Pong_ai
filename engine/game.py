"""
Boucle principale du jeu et gestion des entités.
"""

import pygame
from core.ball import Ball, spawn_ball_left, spawn_ball_right
from config import WIDTH, HEIGHT, GREEN, FPS
from graphics.renderer import draw_background, draw_table, draw_ball, draw_paddle, draw_net
from core.paddle import Paddle
from core.net import Net
from core.table import Table
from engine.collision import check_ball_paddle, check_ball_net, check_table_collision

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ping-Pong 2D - Niveau 1")
        self.clock = pygame.time.Clock()
        self.running = True
        self.net = Net()
        net_center = WIDTH // 2  # Centre du filet
        # Player à gauche: ne peut pas dépasser le centre du filet
        self.player = Paddle(50, HEIGHT//2 - 30, x_min=0, x_max=net_center)
        # Opponent à droite: ne peut pas aller avant le centre du filet
        self.opponent = Paddle(WIDTH - 60, HEIGHT//2 - 30, x_min=net_center, x_max=WIDTH)
        self.players = [self.player, self.opponent]
        self.table = Table()
        
        # Pour afficher la vitesse
        self.font = pygame.font.Font(None, 36)
        self.debug_timer = 0  # compteur pour affichage toutes les 0.5s
        self.last_ball_vel = (0, 0)
        self.last_paddle_vel = (0, 0)
        self.last_spin = 0

        # Balles (vide au départ, appuyer sur R ou T pour lancer)
        self.balls = []

    def run(self):
        """Boucle principale du jeu."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()

    def handle_events(self):
        """Gestion des entrées utilisateur."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # Gestion des touches pour spawn de balle (sur KEYDOWN pour éviter répétition)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.spawn_ball_at_left()
                elif event.key == pygame.K_t:
                    self.spawn_ball_at_right()

        keys = pygame.key.get_pressed()

        # joueur 1

        if keys[pygame.K_o]:
            self.opponent.move_up()
        elif keys[pygame.K_l]:
            self.opponent.move_down()
        else:
            self.opponent.stop_vertical()

        # Mouvement horizontal
        if keys[pygame.K_k]:
            self.opponent.move_left()
        elif keys[pygame.K_m]:
            self.opponent.move_right()
        else:
            self.opponent.stop_horizontal()

        # Rotation
        if keys[pygame.K_i]:
            self.opponent.rotate_left(1)
        if keys[pygame.K_p]:
            self.opponent.rotate_right(1)
        
        # joueur 2
        # Mouvement vertical
        if keys[pygame.K_z]:
            self.player.move_up()
        elif keys[pygame.K_s]:
            self.player.move_down()
        else:
            self.player.stop_vertical()

        # Mouvement horizontal
        if keys[pygame.K_q]:
            self.player.move_left()
        elif keys[pygame.K_d]:
            self.player.move_right()
        else:
            self.player.stop_horizontal()

        # Rotation
        if keys[pygame.K_a]:
            self.player.rotate_left(1)
        if keys[pygame.K_e]:
            self.player.rotate_right(1)

    def spawn_ball_at_left(self):
        """Supprime la balle actuelle et en crée une au bord gauche."""
        self.balls.clear()
        self.balls.append(spawn_ball_left(self.table))

    def spawn_ball_at_right(self):
        """Supprime la balle actuelle et en crée une au bord droit."""
        self.balls.clear()
        self.balls.append(spawn_ball_right(self.table))

    def update(self):
        """Met à jour l'état du jeu (physique, logique)."""
        # Timer pour affichage debug toutes les 0.5s
        self.debug_timer += 1
        if self.debug_timer >= 30:  # 30 frames à 60fps = 0.5s
            self.debug_timer = 0
            if self.balls:
                ball = self.balls[0]
                self.last_ball_vel = (ball.vel[0], ball.vel[1])
                self.last_paddle_vel = (self.player.vel[0], self.player.vel[1])
                self.last_spin = ball.angular_speed
                print(f"Balle: vx={ball.vel[0]:.1f}, vy={ball.vel[1]:.1f} | Raquette: vx={self.player.vel[0]:.1f}, vy={self.player.vel[1]:.1f} | Spin: {ball.angular_speed:.1f}")
        
        for ball in self.balls:
            ball.update()
            # collisions
            check_table_collision(ball, self.table)
            check_ball_net(ball, self.net)
            # collision avec les raquettes
            for player in self.players:
                check_ball_paddle(ball, player, self.screen)
        
        for player in self.players:
            player.update(1.0/FPS)  # dt = 1/120 = 0.0083s par frame

    def draw(self):
        """Dessine les éléments à l'écran."""
        draw_background(self.screen)
        draw_table(self.screen, self.table)
        for ball in self.balls:
            draw_ball(self.screen, ball)
        draw_paddle(self.screen, self.player, (255, 0, 0))
        draw_paddle(self.screen, self.opponent, (0, 0, 0))
        draw_net(self.screen, self.net)
        
        # Affichage debug des vitesses à l'écran
        vel_text = f"Balle: vx={self.last_ball_vel[0]:.0f} vy={self.last_ball_vel[1]:.0f}"
        paddle_text = f"Raquette: vx={self.last_paddle_vel[0]:.0f} vy={self.last_paddle_vel[1]:.0f}"
        spin_text = f"Spin: {self.last_spin:.0f}"
        text_surface = self.font.render(vel_text, True, (255, 255, 255))
        text_surface2 = self.font.render(paddle_text, True, (255, 255, 0))
        text_surface3 = self.font.render(spin_text, True, (0, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        self.screen.blit(text_surface2, (10, 40))
        self.screen.blit(text_surface3, (10, 70))
        
        pygame.display.flip()
