"""
Script d'entraînement PPO pour Ping-Pong.
Style Phil's code - simple et efficace.

Usage:
    python train.py                      # Entraînement (1000 épisodes)
    python train.py --render             # Avec affichage
    python train.py --mode play          # Jouer avec un modèle entraîné
    python train.py --episodes 500       # Nombre d'épisodes personnalisé
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ai.agent import Agent, predict_action
from ai.environment import PingPongEnv


def plot_learning_curve(x, scores, figure_file):
    """Trace la courbe d'apprentissage."""
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file)
    plt.close()
    print(f"Courbe sauvegardée: {figure_file}")


def train(n_games=1000, N=256, batch_size=5, n_epochs=4, alpha=0.0003, 
          render=False, save_best=True):
    """
    Entraîne l'agent PPO sur Ping-Pong.
    
    Args:
        n_games: Nombre d'épisodes d'entraînement
        N: Nombre de steps avant chaque mise à jour
        batch_size: Taille des mini-batches
        n_epochs: Nombre d'epochs par mise à jour
        alpha: Learning rate
        render: Afficher le jeu pendant l'entraînement
        save_best: Sauvegarder le meilleur modèle
    """
    # Créer l'environnement
    render_mode = "human" if render else None
    env = PingPongEnv(render_mode=render_mode)
    
    # Créer l'agent
    # Observation: 12 valeurs, Actions: 3 valeurs continues
    agent = Agent(
        n_actions=3,          # move_x, move_y, rotate
        input_dims=12,        # taille de l'observation
        gamma=0.99,
        alpha=alpha,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=batch_size,
        n_epochs=n_epochs
    )
    
    figure_file = 'plots/pingpong_learning.png'
    
    best_score = float('-inf')
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    print("=== Démarrage de l'entraînement PPO ===")
    print(f"Épisodes: {n_games}, Steps avant update: {N}")
    print(f"Batch size: {batch_size}, Epochs: {n_epochs}, LR: {alpha}")
    print("=" * 50)

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            # Choisir une action
            action, prob, val = agent.choose_action(observation)
            
            # Exécuter l'action
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            n_steps += 1
            score += reward
            
            # Stocker la transition
            agent.remember(observation, action, prob, val, reward, done)
            
            # Apprendre tous les N steps
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # Sauvegarder le meilleur modèle
        if save_best and avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # Afficher la progression
        print(f'Episode {i+1:4d} | Score: {score:7.1f} | '
              f'Avg: {avg_score:7.1f} | Steps: {n_steps:6d} | '
              f'Learn iters: {learn_iters:4d}')
    
    # Tracer la courbe d'apprentissage
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
    env.close()
    print("=== Entraînement terminé ===")
    
    return agent, score_history


def play(model_path='models/ppo', num_episodes=5):
    """
    Joue avec un agent entraîné.
    
    Args:
        model_path: Chemin vers les modèles sauvegardés
        num_episodes: Nombre d'épisodes à jouer
    """
    env = PingPongEnv(render_mode="human")
    
    agent = Agent(
        n_actions=3,
        input_dims=12,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent.load_models()
    
    print("=== Mode Jeu ===")
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Action déterministe pour le jeu
            action = predict_action(agent, observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Ping-Pong Training')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'play'],
                        help='Mode: train ou play')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Nombre d\'épisodes pour l\'entraînement')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu pendant l\'entraînement')
    parser.add_argument('--model_path', type=str, default='models/ppo',
                        help='Chemin vers le modèle')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(n_games=args.episodes, render=args.render) # render=args.render
    else:
        play(model_path=args.model_path, num_episodes=5)
