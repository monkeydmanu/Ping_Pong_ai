"""
Classe représentant le filet.
"""

from config import WIDTH, HEIGHT

class Net:
    def __init__(self, width=40, height=100): # w = 4 et h = 60
        self.width = width
        self.height = height
        self.x = WIDTH // 2 - width //2
        self.y = HEIGHT - height - 50 # filet sur la table

    def get_rect(self):
        return (self.x, self.y, self.width, self.height)