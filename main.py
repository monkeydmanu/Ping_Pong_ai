"""
Point d'entrée du jeu de ping-pong.
"""

from engine.game import Game
    
if __name__ == "__main__":
    game = Game()
    game.run()

# il faut que la vitesse de la balle tienne compte de la taille de l'écran
# pas de cloquage pour les raquettes en mousses, on garde la formule pour la phase noir de restitution en fonction des m/s, on peut commencer avec la gravité
# il faut vérifier avec la formule énergie loss de wikipédia qu'on respecte environ la même valeur peu importe la collision
# il faudra changer les restitutions en fonction de la vitesse de la surface qui touche la balle