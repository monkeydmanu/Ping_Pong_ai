"""
Classe représentant la table de ping-pong.
"""

from config import WIDTH, HEIGHT, TABLE_Y

class Table:
    def __init__(self, width=None, height=None):
        # Largeur et hauteur de la table
        self.width = width if width is not None else WIDTH * 0.8  # 80% de l'écran par défaut
        self.height = height if height is not None else 30       # hauteur fixe par défaut
        # Position pour centrer horizontalement
        self.x = (WIDTH - self.width) / 2
        # Position verticale sur l'écran
        self.y = TABLE_Y

    def get_rect(self):
        """Retourne la position et taille sous forme de tuple (x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)