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
        self.score_font = pygame.font.Font(None, 72)  # Police plus grande pour le score
        self.debug_timer = 0  # compteur pour affichage toutes les 0.5s
        self.last_ball_vel = (0, 0)
        self.last_paddle_vel = (0, 0)
        self.last_spin = 0
        
        # Scores des joueurs
        self.score_left = 0   # Score du joueur de gauche
        self.score_right = 0  # Score du joueur de droite
        self.serving_side = 'left'  # Qui sert ('left' ou 'right')
        self.point_message = ""  # Message à afficher (ex: "Faute!")
        self.message_timer = 0   # Timer pour afficher le message

        # Balles (vide au départ, appuyer sur R ou T pour lancer)
        self.balls = []
        
        # Tracking du côté de la balle (pour réinitialiser can_hit)
        # 'left' si la balle est à gauche du filet, 'right' si à droite
        self.ball_side = None

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
        # Timer pour message temporaire
        if self.message_timer > 0:
            self.message_timer -= 1
        
        # Timer pour affichage debug toutes les 0.5s
        self.debug_timer += 1
        if self.debug_timer >= 30:  # 30 frames à 60fps = 0.5s
            self.debug_timer = 0
            if self.balls:
                ball = self.balls[0]
                self.last_ball_vel = (ball.vel[0], ball.vel[1])
                self.last_paddle_vel = (self.player.vel[0], self.player.vel[1])
                self.last_spin = ball.angular_speed
        
        for ball in self.balls[:]:
            ball.update()
            
            # Détecter le changement de côté de la balle
            net_center = WIDTH // 2
            current_side = 'left' if ball.pos[0] < net_center else 'right'
            
            # Si la balle change de côté, réinitialiser can_hit et les compteurs de rebonds
            if self.ball_side is not None and current_side != self.ball_side:
                # Vérifier si le service est valide AVANT de reset
                if ball.is_service:
                    # Le serveur à gauche doit avoir fait rebondir sur son côté (gauche) avant de passer à droite
                    if ball.last_hit_by == 'left' and self.ball_side == 'left' and ball.bounces_left == 0:
                        self.award_point('right', "Service invalide!")
                        self.balls.remove(ball)
                        continue
                    # Le serveur à droite doit avoir fait rebondir sur son côté (droite) avant de passer à gauche
                    elif ball.last_hit_by == 'right' and self.ball_side == 'right' and ball.bounces_right == 0:
                        self.award_point('left', "Service invalide!")
                        self.balls.remove(ball)
                        continue
                    ball.is_service = False
                
                for player in self.players:
                    player.can_hit = True
                # Reset le compteur de rebonds du côté qu'on quitte
                if self.ball_side == 'left':
                    ball.bounces_left = 0
                else:
                    ball.bounces_right = 0
            
            self.ball_side = current_side
            
            # collisions
            check_table_collision(ball, self.table)
            check_ball_net(ball, self.net)
            
            # Vérifier double rebond (faute)
            point_scored = self.check_point_scored(ball)
            if point_scored:
                continue
            
            # collision avec les raquettes
            for i, player in enumerate(self.players):
                old_can_hit = player.can_hit
                check_ball_paddle(ball, player, self.screen)
                # Si le joueur a frappé la balle
                if old_can_hit and not player.can_hit:
                    ball.last_hit_by = 'left' if i == 0 else 'right'
        
        for player in self.players:
            player.update(1.0/FPS)  # dt = 1/120 = 0.0083s par frame
    
    def check_point_scored(self, ball):
        """Vérifie si un point est marqué et met à jour le score."""
        point_for = None  # 'left' ou 'right'
        reason = ""
        
        # Double rebond sur le même côté = point pour l'adversaire
        if ball.bounces_left >= 2:
            point_for = 'right'
            reason = "Double rebond!"
        elif ball.bounces_right >= 2:
            point_for = 'left'
            reason = "Double rebond!"
        
        # Balle sortie par la gauche
        elif ball.pos[0] < 0:
            point_for = 'right'
            reason = "Sortie!"
        
        # Balle sortie par la droite
        elif ball.pos[0] > WIDTH:
            point_for = 'left'
            reason = "Sortie!"
        
        # Balle sortie par le bas (tombe)
        elif ball.pos[1] > HEIGHT:
            if ball.last_hit_by == 'left':
                point_for = 'right'
            else:
                point_for = 'left'
            reason = "Dans le filet!" if ball.pos[0] > WIDTH//2 - 50 and ball.pos[0] < WIDTH//2 + 50 else "Faute!"
        
        if point_for:
            self.award_point(point_for, reason)
            self.balls.remove(ball)
            return True
        
        return False
    
    def award_point(self, winner, reason):
        """Attribue un point au gagnant."""
        if winner == 'left':
            self.score_left += 1
        else:
            self.score_right += 1
        
        self.point_message = reason
        self.message_timer = 120  # 2 secondes à 60 fps
        
        # Alterner le service tous les 2 points
        total_points = self.score_left + self.score_right
        if total_points % 2 == 0:
            self.serving_side = 'left' if self.serving_side == 'right' else 'right'

    def draw(self):
        """Dessine les éléments à l'écran."""
        draw_background(self.screen)
        draw_table(self.screen, self.table)
        for ball in self.balls:
            draw_ball(self.screen, ball)
        draw_paddle(self.screen, self.player, (255, 0, 0))
        draw_paddle(self.screen, self.opponent, (0, 0, 0))
        draw_net(self.screen, self.net)
        
        # === AFFICHAGE DU SCORE ===
        self.draw_score()
        
        # Affichage debug des vitesses à l'écran (en bas)
        vel_text = f"Balle: vx={self.last_ball_vel[0]:.0f} vy={self.last_ball_vel[1]:.0f}"
        paddle_text = f"Raquette: vx={self.last_paddle_vel[0]:.0f} vy={self.last_paddle_vel[1]:.0f}"
        spin_text = f"Spin: {self.last_spin:.0f}"
        text_surface = self.font.render(vel_text, True, (255, 255, 255))
        text_surface2 = self.font.render(paddle_text, True, (255, 255, 0))
        text_surface3 = self.font.render(spin_text, True, (0, 255, 255))
        self.screen.blit(text_surface, (10, HEIGHT - 90))
        self.screen.blit(text_surface2, (10, HEIGHT - 60))
        self.screen.blit(text_surface3, (10, HEIGHT - 30))
        
        pygame.display.flip()
    
    def draw_score(self):
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
        
        # Indicateur de service (petite balle à côté du serveur)
        service_x = WIDTH // 2 - 80 if self.serving_side == 'left' else WIDTH // 2 + 80
        pygame.draw.circle(self.screen, (255, 200, 0), (service_x, 40), 8)
        pygame.draw.circle(self.screen, (255, 140, 0), (service_x, 40), 6)
        
        # Labels des joueurs
        left_label = self.font.render("Joueur 1", True, (255, 100, 100))
        right_label = self.font.render("Joueur 2", True, (100, 100, 100))
        self.screen.blit(left_label, (WIDTH // 2 - 180, 60))
        self.screen.blit(right_label, (WIDTH // 2 + 80, 60))
        
        # Message temporaire (ex: "Double rebond!")
        if self.message_timer > 0 and self.point_message:
            # Effet de fade out
            alpha = min(255, self.message_timer * 4)
            msg_surface = self.score_font.render(self.point_message, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(WIDTH // 2, 120))
            self.screen.blit(msg_surface, msg_rect)
