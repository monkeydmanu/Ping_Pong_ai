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
from collections import deque

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


def plot_episode_rewards(rewards, episode_num, save_dir='plots'):
    """Trace les rewards step par step d'un épisode."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Rewards à chaque step
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'b-', alpha=0.7, linewidth=0.8)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title(f'Rewards par step - Episode {episode_num}')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Rewards cumulées
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(rewards)
    plt.plot(cumulative, 'g-', linewidth=1.5)
    plt.title(f'Reward cumulée - Episode {episode_num}')
    plt.xlabel('Step')
    plt.ylabel('Reward cumulée')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'episode_{episode_num}_rewards.png')
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"\n📊 Plot sauvegardé: {filename}")
    
    # Afficher des stats
    print(f"   Steps: {len(rewards)} | Total: {sum(rewards):.2f} | ")
    print(f"   Min: {min(rewards):.2f} | Max: {max(rewards):.2f} | Mean: {np.mean(rewards):.4f}")


def setup_live_plot():
    """Configure le plot live pour l'entraînement."""
    plt.ion()  # Mode interactif
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.set_title('Score par épisode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Moyenne glissante (100 épisodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score moyen')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax1, ax2


def update_live_plot(fig, ax1, ax2, scores, update_freq=10):
    """Met à jour le plot live."""
    if len(scores) % update_freq != 0:
        return
    
    ax1.clear()
    ax2.clear()
    
    x = list(range(1, len(scores) + 1))
    
    # Scores bruts
    ax1.plot(x, scores, 'b-', alpha=0.5, linewidth=0.5)
    ax1.set_title('Score par épisode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # Moyenne glissante
    if len(scores) > 0:
        running_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
        ax2.plot(x, running_avg, 'g-', linewidth=2)
        ax2.set_title(f'Moyenne glissante (100 ép.) - Actuel: {running_avg[-1]:.1f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score moyen')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


def train(n_games=1000, N=256, batch_size=5, n_epochs=4, alpha=0.0003, 
          render=False, save_best=True, live_plot=True, plot_first_episode=True,
          resume=False, model_path='models/ppo'):
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
        live_plot: Afficher un graphique en temps réel
        plot_first_episode: Sauvegarder le plot des rewards du premier épisode
        resume: Reprendre l'entraînement depuis le dernier modèle sauvegardé
        model_path: Chemin vers le modèle à charger/sauvegarder
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
        n_epochs=n_epochs,
        chkpt_dir=model_path
    )
    
    # Charger le modèle existant si resume=True
    if resume:
        actor_path = os.path.join(model_path, 'actor_torch_ppo')
        if os.path.exists(actor_path):
            agent.load_models()
            print(f"✅ Modèle chargé depuis {model_path}")
        else:
            print(f"⚠️ Aucun modèle trouvé dans {model_path}, démarrage from scratch")
    
    figure_file = 'plots/pingpong_learning.png'
    
    best_score = float('-inf')
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    # Setup live plot
    fig, ax1, ax2 = None, None, None
    if live_plot:
        try:
            fig, ax1, ax2 = setup_live_plot()
        except:
            print("⚠️ Impossible d'activer le plot live (pas de display)")
            live_plot = False

    print("=== Démarrage de l'entraînement PPO ===")
    print(f"Mode: {'RESUME' if resume else 'NOUVEAU'}")
    print(f"Épisodes: {n_games}, Steps avant update: {N}")
    print(f"Batch size: {batch_size}, Epochs: {n_epochs}, LR: {alpha}")
    print("=" * 50)

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        episode_rewards = []  # Track rewards de cet épisode
        episode_hits = 0
        
        while not done:
            # Choisir une action
            action, prob, val = agent.choose_action(observation)
            
            # Exécuter l'action
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            n_steps += 1
            score += reward
            episode_rewards.append(reward)
            
            # Récupérer le nombre de hits depuis l'environnement
            episode_hits = info.get('agent_hits', 0)
            
            # Stocker la transition
            agent.remember(observation, action, prob, val, reward, done)
            
            # Apprendre tous les N steps
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            
            observation = observation_
        
        # Plot des rewards du premier épisode
        if i == 0 and plot_first_episode:
            plot_episode_rewards(episode_rewards, episode_num=1)
        
        # Plot tous les 10 épisodes pour débuguer
        if (i + 1) % 100 == 0:
            plot_episode_rewards(episode_rewards, episode_num=i+1)
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # Sauvegarder le meilleur modèle
        if save_best and avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        # Déterminer si l'agent a gagné
        won = "✓" if score > 10 else "✗"

        # Afficher la progression
        print(f'Ep {i+1:4d} | Score: {score:7.1f} | Avg: {avg_score:7.1f} | '
              f'Hits: {episode_hits} | Won: {won} | Steps: {len(episode_rewards):3d}')
        
        # Mettre à jour le plot live
        if live_plot and fig is not None:
            update_live_plot(fig, ax1, ax2, score_history, update_freq=10)
    
    # Fermer le plot interactif
    if live_plot:
        plt.ioff()
        plt.close('all')
    
    # Tracer la courbe d'apprentissage finale
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
    # Vérifier que le modèle existe
    actor_path = os.path.join(model_path, 'actor_torch_ppo')
    if not os.path.exists(actor_path):
        print(f"❌ Erreur: Aucun modèle trouvé dans {model_path}")
        print("   Lance d'abord l'entraînement avec: python train.py")
        return
    
    env = PingPongEnv(render_mode="human")
    
    agent = Agent(
        n_actions=3,
        input_dims=12,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent.load_models()
    print(f"✅ Modèle chargé depuis {model_path}")
    
    print("=== Mode Jeu ===")
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        hits = 0
        
        while not done:
            # Action déterministe pour le jeu
            action = predict_action(agent, observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            hits = info.get('agent_hits', 0)
            done = terminated or truncated
        
        won = "✓ Gagné" if total_reward > 10 else "✗ Perdu"
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f} | Hits: {hits} | {won}")
    
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
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre l\'entraînement depuis le dernier modèle sauvegardé')
    parser.add_argument('--model_path', type=str, default='models/ppo',
                        help='Chemin vers le modèle')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(n_games=args.episodes, render=args.render, 
              resume=True, model_path=args.model_path)  # Toujours charger le modèle s'il existe
    else:
        play(model_path=args.model_path, num_episodes=5)
