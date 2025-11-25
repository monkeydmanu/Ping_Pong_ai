"""
Boucle principale du jeu et gestion des entités.
"""

import pygame
from core.ball import Ball
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
        self.player = Paddle(50, HEIGHT//2 - 30)      # raquette gauche
        self.opponent = Paddle(WIDTH - 60, HEIGHT//2 - 30)  # raquette droite
        self.players = [self.player, self.opponent]
        self.net = Net()
        self.table = Table()

        # Balles de test
        x_table, y_table, w_table, h_table = self.table.get_rect()
        self.balls = [
            #Ball(x=20, y=y_table + h_table - 10, vx=200, vy=0, angular_speed=-300),
            Ball(x=300, y=y_table - 200, vx=0, vy=0, angular_speed=500),
            #Ball(x=x_table - 4, y=y_table - 200, vx=0, vy=-200, angular_speed=100),
            #Ball(x=x_table - 9, y=y_table - 200, vx=0, vy=-200, angular_speed=100),              # coin gauche
            #Ball(x=x_table - 5, y=y_table - 100, vx=0, vy=-150, angular_speed=-100),          # proche coin gauche
            #Ball(x=x_table + w_table + 5, y=y_table - 150, vx=0, vy=-200, angular_speed=-100),   # coin droit
            #Ball(x=x_table + w_table + 5, y=y_table - 120, vx=0, vy=-150, angular_speed=-100) # proche coin droit
        ]

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

        keys = pygame.key.get_pressed()

        # joueur 1
        if keys[pygame.K_UP]:
            self.opponent.move_up()
        if keys[pygame.K_DOWN]:
            self.opponent.move_down()
        
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

    def update(self):
        """Met à jour l'état du jeu (physique, logique)."""
        for ball in self.balls:
            ball.update()
            # collisions
            check_table_collision(ball, self.table)
            check_ball_net(ball, self.net)
        for player in self.players:
            check_ball_paddle(ball, player, self.screen)
            player.update(FPS/60)

    def draw(self):
        """Dessine les éléments à l’écran."""
        draw_background(self.screen)
        draw_table(self.screen, self.table)
        for ball in self.balls:
            draw_ball(self.screen, ball)
        draw_paddle(self.screen, self.player, (255, 0, 0))
        draw_paddle(self.screen, self.opponent, (0, 0, 0))
        draw_net(self.screen, self.net)
        pygame.display.flip()
