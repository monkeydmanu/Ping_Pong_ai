"""
Script pour entraîner l'IA Ping-Pong.

Usage:
    python train.py              # Entraînement sans rendu (rapide)
    python train.py --render     # Entraînement avec rendu (lent mais visuel)
    python train.py --play       # Jouer avec un modèle entraîné
"""

import argparse
from ai.agent import train_agent, play_with_agent, PingPongAgent
from ai.environment import PingPongEnv


def main():
    parser = argparse.ArgumentParser(description="Entraînement IA Ping-Pong")
    parser.add_argument("--render", action="store_true", 
                        help="Afficher le jeu pendant l'entraînement")
    parser.add_argument("--play", action="store_true",
                        help="Jouer avec un modèle entraîné")
    parser.add_argument("--model", type=str, default="models/ppo_pingpong",
                        help="Chemin du modèle")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Nombre de timesteps d'entraînement")
    parser.add_argument("--envs", type=int, default=1,
                        help="Nombre d'environnements parallèles")
    
    args = parser.parse_args()
    
    if args.play:
        # Mode jeu avec modèle entraîné
        print("=== Mode Jeu ===")
        play_with_agent(args.model, num_episodes=10)
    else:
        # Mode entraînement
        print("=== Mode Entraînement ===")
        print(f"Timesteps: {args.timesteps}")
        print(f"Environnements: {args.envs}")
        print(f"Rendu: {args.render}")
        
        agent = train_agent(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            render=args.render
        )
        
        print("\n=== Entraînement terminé ===")
        print(f"Modèle sauvegardé dans: models/ppo_pingpong.zip")


if __name__ == "__main__":
    main()
