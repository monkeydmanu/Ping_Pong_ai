"""
Point d'entrée du jeu de ping-pong.
"""

from engine.game import Game
    
if __name__ == "__main__":
    game = Game()
    game.run()

# il faut que la vitesse de la balle tienne compte de la taille de l'écran
# pas de cloquage pour les raquettes en mousses, on garde la formule pour la phase noir de restitution en fonction des m/s