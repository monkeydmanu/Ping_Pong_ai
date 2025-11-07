"""
Classe représentant la table de ping-pong.
"""

from config import WIDTH, HEIGHT, TABLE_Y, TABLE_WIDTH_PX, TABLE_HEIGHT_PX, PIXELS_PER_METER

class Table:
    def __init__(self, width=TABLE_WIDTH_PX, height=TABLE_HEIGHT_PX):
        # Largeur et hauteur de la table
        self.width = width
        self.height = height
        # Position pour centrer horizontalement
        self.x = WIDTH // 2 - width //2
        # Position verticale sur l'écran
        self.y = TABLE_Y

    def get_rect(self):
        """Retourne la position et taille sous forme de tuple (x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)