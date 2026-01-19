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
import pygame
from collections import deque
from config import FPS

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


def train(n_games=1000, N=2048, batch_size=64, n_epochs=10, alpha=0.0003, 
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
    # Observation: 18 valeurs, Actions: 3 valeurs continues
    agent = Agent(
        n_actions=3,          # move_x, move_y, rotate
        input_dims=18,        # taille de l'observation (18 variables)
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
            observation_, terminated, info = env.step(action)
            
            # Calculer la récompense pour l'entraînement
            reward = env.compute_reward(info)
            done = terminated
            
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
        
        # Déterminer si l'agent a gagné (utiliser le flag point_winner_side)
        episode_winner = env.point_winner_side
        if episode_winner == 'left' and env.agent_side == 'left':
            won = "✓"
        elif episode_winner == 'right' and env.agent_side == 'right':
            won = "✓"
        else:
            won = "✗"

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


def play_ai_vs_ai(model_path='models/ppo', num_episodes=5):
    """
    IA vs IA avec affichage visuel (évaluation sans entraînement).
    
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
    
    # Importer Game uniquement quand nécessaire
    from engine.game import Game
    
    agent = Agent(
        n_actions=3,
        input_dims=18,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent.load_models()
    print(f"✅ Modèle chargé depuis {model_path}")
    
    print("=== Mode Jeu IA vs IA ===")
    print("Les deux raquettes sont contrôlées par des IA")
    
    game = Game(player1_type="ai", player2_type="ai")
    
    for episode in range(num_episodes):
        game.env.reset()
        game.score_left = 0
        game.score_right = 0
        
        steps = 0
        done = False
        
        while not done and game.running and steps < 3000:
            # Gestion des events pour garder la fenêtre réactive (fermeture possible)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    game.running = False
                    break

            if not game.running:
                break

            # IA gauche (agent_paddle) - utilise le modèle entraîné
            obs = game.env._get_observation()
            action_p1 = predict_action(agent, obs, deterministic=True)
            
            # IA droite (opponent_paddle) - IA simple intégrée
            action_p2 = game.env._get_opponent_action()
            
            # Simuler directement via env (pas de reward car pas d'entraînement)
            obs, done, info = game.env.step(action_p1, action_p2)
            game.score_left = info.get('score_left', 0)
            game.score_right = info.get('score_right', 0)
            game.point_message = info.get('point_message', '')
            
            steps += 1
            
            # Afficher visuellement
            game.draw()
            game.clock.tick(FPS)
        
        if not game.running:
            break
        
        print(f"Episode {episode + 1}: {game.score_left} - {game.score_right}")
    
    game.running = False
    pygame.quit()


def play_human_vs_human():
    """
    1v1 entre deux joueurs humains avec affichage complet.
    Joueur 1 (gauche): Z/S (vertical), Q/D (horizontal), A/E (rotation)
    Joueur 2 (droite): O/L (vertical), K/M (horizontal), I/P (rotation)
    """
    from engine.game import Game
    
    print("=== Mode Humain vs Humain ===")
    print("Joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
    print("Joueur 2 (droite): O/L=vertical, K/M=horizontal, I/P=rotation")
    
    game = Game(player1_type="human", player2_type="human")
    game.run()
    
    print("Fin du jeu!")


def play_ai_vs_human(model_path='models/ppo'):
    """
    IA vs Joueur Humain avec affichage complet.
    
    Args:
        model_path: Chemin vers les modèles sauvegardés
    """
    # Vérifier que le modèle existe
    actor_path = os.path.join(model_path, 'actor_torch_ppo')
    if not os.path.exists(actor_path):
        print(f"❌ Erreur: Aucun modèle trouvé dans {model_path}")
        print("   Lance d'abord l'entraînement avec: python train.py")
        return
    
    from engine.game import Game
    
    agent = Agent(
        n_actions=3,
        input_dims=18,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent.load_models()
    print(f"✅ Modèle chargé depuis {model_path}")
    
    print("=== Mode IA vs Humain ===")
    print("Vous êtes le joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
    print("L'IA est le joueur 2 (droite)")
    
    game = Game(player1_type="human", player2_type="ai")
    
    # On doit modifier le jeu pour utiliser l'agent IA pour player2
    # Créer une boucle spéciale
    while game.running:
        # Garde la fenêtre réactive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
        if not game.running:
            break

        # Action du joueur humain (déjà récupérée par handle_events)
        game.handle_events()
        if not game.running:
            break
        action_p1 = game.action_p1
        
        # Action de l'IA
        obs = game.env._get_observation()
        action_p2 = predict_action(agent, obs, deterministic=True)
        
        # Mettre à jour l'env (pas de reward car pas d'entraînement)
        obs, terminated, info = game.env.step(action_p1, action_p2)
        game.score_left = info.get('score_left', 0)
        game.score_right = info.get('score_right', 0)
        
        if terminated:
            game.point_message = info.get('point_message', '')
            game.message_timer = 120
            game.env.reset()
        
        # Décrementer timer
        if game.message_timer > 0:
            game.message_timer -= 1
        
        game.draw()
        game.clock.tick(FPS)
    
    pygame.quit()
    print("Fin du jeu!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Ping-Pong Training')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'play', 'human', 'ai_vs_human'],
                        help='Modes: train (IA vs IA entraînement), play (IA vs IA affiché), human (humain vs humain), ai_vs_human (IA vs humain)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Nombre d\'épisodes pour l\'entraînement')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu pendant l\'entraînement')
    parser.add_argument('--render_plot', action='store_true',
                        help='Afficher les graphiques en temps réel')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Reprendre l\'entraînement depuis le dernier modèle sauvegardé (par défaut: True)')
    parser.add_argument('--fresh', action='store_true',
                        help='Démarrer un nouvel entraînement from scratch (ignore le modèle existant)')
    parser.add_argument('--model_path', type=str, default='models/ppo',
                        help='Chemin vers le modèle')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Par défaut resume=True, sauf si --fresh est spécifié
        should_resume = args.resume and not args.fresh
        train(n_games=args.episodes, render=args.render, live_plot=args.render_plot,
              resume=should_resume, model_path=args.model_path)
    elif args.mode == 'play':
        play_ai_vs_ai(model_path=args.model_path, num_episodes=args.episodes)
    elif args.mode == 'human':
        play_human_vs_human()
    elif args.mode == 'ai_vs_human':
        play_ai_vs_human(model_path=args.model_path)

# pour train ia vs ia
# python train.py --mode train --episodes 1000 --render_plot

# pour play ia vs ia 
# python train.py --mode play --model_path models/ppo --episodes 20

# pour humain vs humain
# python train.py --mode human

# pour ia vs humain
# python train.py --mode ai_vs_human --model_path models/ppo