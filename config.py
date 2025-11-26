"""
Fichier de configuration — toutes les constantes du jeu.
"""

# --- Fenêtre ---
WIDTH, HEIGHT = 1200, 800
TABLE_Y = HEIGHT - 150

TABLE_REAL_LENGTH = 2.74  # m
TABLE_REAL_HEIGHT = 0.1 # m

COEF_TAILLE_TABLE = 0.7


PIXELS_PER_METER = (WIDTH * COEF_TAILLE_TABLE) / TABLE_REAL_LENGTH

TABLE_WIDTH_PX = (TABLE_REAL_LENGTH * PIXELS_PER_METER) * COEF_TAILLE_TABLE
TABLE_HEIGHT_PX = (TABLE_REAL_HEIGHT * PIXELS_PER_METER) * COEF_TAILLE_TABLE

NET_REAL_HEIGHT = 0.1525      # m
NET_REAL_WIDTH = 0.05      # m

RACKET_REAL_HEIGHT = 0.2    # m    en vrai c'est 0.157 m mais on va agrandir un peu
RACKET_REAL_WIDTH = 0.05     # m    en vrai c'est 0.02 m mais on va agrandir un peu

NET_HEIGHT_PX = NET_REAL_HEIGHT * PIXELS_PER_METER
NET_WIDTH_PX = NET_REAL_WIDTH * PIXELS_PER_METER

RACKET_HEIGHT_PX = RACKET_REAL_HEIGHT * PIXELS_PER_METER
RACKET_WIDTH_PX = RACKET_REAL_WIDTH * PIXELS_PER_METER


BALL_RADIUS = 10


# --- Physique ---
GRAVITY = 9.81          # gravité (m/s²)
RESTITUTION = 0.9       # perte d'énergie au rebond
FPS = 120

# --- Effet Magnus (balle de ping-pong) ---
BALL_MASS = 0.0027          # masse de la balle (kg) — 2.7g
BALL_REAL_RADIUS = 0.02     # rayon réel (m) — 20mm de diamètre = 40mm
AIR_DENSITY = 1.2           # densité de l'air (kg/m³)
MAGNUS_COEFFICIENT = 0.5    # coefficient de lift (Cl) — empirique, ajustable
DRAG_COEFFICIENT = 0.4      # coefficient de traînée (Cd) — pour une sphère lisse

# --- Constante de ping pong de vitesse utile --- 
V_MAX_REEL = 120 # m/s
VPX_FRAME_MAX = V_MAX_REEL * PIXELS_PER_METER / FPS
SPEED_RACKET = 1.5 * PIXELS_PER_METER / FPS

# --- Couleurs (R, G, B) ---
GREEN = (40, 150, 60)
RED = (200, 40, 40)
BROWN = (80, 50, 30)
PINGPONG_ORANGE = (255, 140, 0)
