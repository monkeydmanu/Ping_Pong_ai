"""
Classe représentant le filet.
"""

from config import WIDTH, HEIGHT, NET_HEIGHT_PX, TABLE_Y

class Net:
    def __init__(self, width=20, height=NET_HEIGHT_PX): # w = 4 et h = 60
        self.width = width
        self.height = height
        self.x = WIDTH // 2 - width //2
        self.y = TABLE_Y - height # filet sur la table

    def get_rect(self):
        return (self.x, self.y, self.width, self.height)