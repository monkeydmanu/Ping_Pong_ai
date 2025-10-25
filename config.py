"""
Fichier de configuration — toutes les constantes du jeu.
"""

# --- Fenêtre ---
WIDTH, HEIGHT = 1200, 800
TABLE_Y = HEIGHT - 150

TABLE_REAL_LENGTH = 2.74  # m
COEF_TAILLE_TABLE = 0.6


PIXELS_PER_METER = (WIDTH * COEF_TAILLE_TABLE) / TABLE_REAL_LENGTH

NET_REAL_HEIGHT = 0.1525      # m
RACKET_REAL_HEIGHT = 0.2    # m    en vrai c'est 0.157 m mais on va agrandir un peu
RACKET_REAL_WIDTH = 0.05     # m    en vrai c'est 0.02 m mais on va agrandir un peu

NET_HEIGHT_PX = NET_REAL_HEIGHT * PIXELS_PER_METER

RACKET_HEIGHT_PX = RACKET_REAL_HEIGHT * PIXELS_PER_METER
RACKET_WIDTH_PX = RACKET_REAL_WIDTH * PIXELS_PER_METER


BALL_RADIUS = 10

# --- Physique ---
GRAVITY = 9.81          # gravité (m/s²)
RESTITUTION = 0.9       # perte d'énergie au rebond
FPS = 60

# --- Constante de ping pong de vitesse utile --- 
V_MAX_REEL = 120 # m/s
VPX_FRAME_MAX = V_MAX_REEL * PIXELS_PER_METER / FPS

# --- Couleurs (R, G, B) ---
GREEN = (40, 150, 60)
RED = (200, 40, 40)
BROWN = (80, 50, 30)
PINGPONG_ORANGE = (255, 140, 0)
