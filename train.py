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
import pickle
import json
from collections import deque
from config import FPS

from ai.agent import Agent, predict_action
from ai.environment import PingPongEnv

# Pygame sera importé seulement si nécessaire
try:
    import pygame
except ImportError:
    pygame = None


def save_training_state(checkpoint_dir, episode, all_rewards, all_entropy, all_std, 
                        all_critic_loss, all_values, score_history, best_score, training_phase=0, 
                        learn_call_count=0):
    """Sauvegarde l'état complet de l'entraînement en mémoire."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        'episode': episode,  # Dernier épisode complété
        'all_rewards': all_rewards,
        'all_entropy': all_entropy,
        'all_std': all_std,
        'all_critic_loss': all_critic_loss,
        'all_values': all_values,  # Mean critic values per learn
        'score_history': score_history,
        'best_score': best_score,
        'training_phase': training_phase,  # Phase de curriculum
        'learn_call_count': learn_call_count,  # Nombre d'appels à learn()
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, 'training_state.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"✅ État d'entraînement sauvegardé: {checkpoint_file}")
    return checkpoint_file


def load_training_state(checkpoint_dir):
    """Charge l'état complet de l'entraînement."""
    checkpoint_file = os.path.join(checkpoint_dir, 'training_state.pkl')
    
    if not os.path.exists(checkpoint_file):
        print(f"⚠️  Aucun état d'entraînement trouvé: {checkpoint_file}")
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        print(f"✅ État d'entraînement chargé: {checkpoint_file}")
        print(f"   Reprise à partir de l'épisode {state['episode'] + 1}")
        return state
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'état: {e}")
        return None


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


def plot_century_metrics(episode_range, rewards_history, entropy_history, 
                         std_history, critic_loss_history, save_dir='plots'):
    """Trace toutes les métriques pour un siècle (100 épisodes).
    
    Args:
        episode_range: tuple (start_ep, end_ep) pour le titre
        rewards_history: liste des rewards totaux par épisode
        entropy_history: liste des entropies moyennes par épisode
        std_history: dict avec 'move_x', 'move_y', 'rotation' - listes de std moyens
        critic_loss_history: liste des critic loss par learn
    """
    os.makedirs(save_dir, exist_ok=True)
    
    start_ep, end_ep = episode_range
    episodes = list(range(start_ep, end_ep + 1))
    
    # Créer une figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Métriques d\'entraînement - Episodes {start_ep} à {end_ep}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Rewards par épisode
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards_history, 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Reward Moyen par Step (par épisode)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward Moyen / Step')
    ax1.grid(True, alpha=0.3)
    
    # 2. Entropy (mesure d'exploration) - LEARN CALLS en abscisse
    ax2 = axes[0, 1]
    learn_calls = list(range(1, len(entropy_history) + 1))
    ax2.plot(learn_calls, entropy_history, 'g-', alpha=0.7, linewidth=1)
    ax2.set_title('Entropy par Learn Call', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Learn Call')
    ax2.set_ylabel('Entropy')
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard Deviation par action - LEARN CALLS en abscisse
    ax3 = axes[1, 0]
    ax3.plot(learn_calls, std_history['move_x'], 'r-', alpha=0.7, linewidth=1, label='Move X')
    ax3.plot(learn_calls, std_history['move_y'], 'b-', alpha=0.7, linewidth=1, label='Move Y')
    ax3.plot(learn_calls, std_history['rotation'], 'orange', alpha=0.7, linewidth=1, label='Rotation')
    ax3.set_title('Std par Learn Call', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Learn Call')
    ax3.set_ylabel('Standard Deviation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Critic Loss - LEARN CALLS en abscisse
    ax4 = axes[1, 1]
    ax4.plot(learn_calls, critic_loss_history, 'purple', alpha=0.7, linewidth=1)
    ax4.set_title('Critic Loss par Learn Call', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Learn Call')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'century_{start_ep}_{end_ep}.png')
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"\n📊 Métriques du siècle sauvegardées: {filename}")


def plot_final_summary(all_rewards, all_entropy, all_std, all_critic_loss, all_values, save_dir='plots'):
    """Trace le bilan final avec rewards par épisode et métriques par learn.
    
    Args:
        all_rewards: liste des rewards totaux par épisode (valeurs réelles)
        all_entropy: liste des entropies par learn
        all_std: dict avec 'move_x', 'move_y', 'rotation' - listes de std par learn
        all_critic_loss: liste des critic loss par learn
        all_values: liste des valeurs moyennes d'état par learn
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_episodes = len(all_rewards)
    episodes = list(range(1, n_episodes + 1))
    
    # Créer une figure avec 5 subplots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f'BILAN FINAL - Tous les épisodes (1 à {n_episodes})', 
                 fontsize=18, fontweight='bold')
    
    # 1. Rewards par épisode (moyenne par step)
    ax1 = axes[0, 0]
    ax1.plot(episodes, all_rewards, 'b-', alpha=0.6, linewidth=0.8)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Reward Moyen par Step (par épisode)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Reward Moyen / Step', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Ajouter stats textuelles
    mean_reward = np.mean(all_rewards)
    max_reward = np.max(all_rewards)
    min_reward = np.min(all_rewards)
    ax1.text(0.02, 0.98, f'Mean: {mean_reward:.1f}\nMax: {max_reward:.1f}\nMin: {min_reward:.1f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 2. Entropy (mesure d'exploration) - LEARN CALLS en abscisse
    ax2 = axes[0, 1]
    learn_calls = list(range(1, len(all_entropy) + 1))
    ax2.plot(learn_calls, all_entropy, 'g-', alpha=0.6, linewidth=0.8)
    ax2.set_title('Entropy par Learn Call (exploration)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Learn Call', fontsize=11)
    ax2.set_ylabel('Entropy', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Stats
    mean_entropy = np.mean(all_entropy)
    final_entropy = all_entropy[-1] if all_entropy else 0
    ax2.text(0.02, 0.98, f'Mean: {mean_entropy:.3f}\nFinal: {final_entropy:.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=9)
    
    # 3. Standard Deviation par action - LEARN CALLS en abscisse
    ax3 = axes[1, 0]
    ax3.plot(learn_calls, all_std['move_x'], 'r-', alpha=0.6, linewidth=0.8, label='Move X')
    ax3.plot(learn_calls, all_std['move_y'], 'b-', alpha=0.6, linewidth=0.8, label='Move Y')
    ax3.plot(learn_calls, all_std['rotation'], 'orange', alpha=0.6, linewidth=0.8, label='Rotation')
    ax3.set_title('Std par Learn Call (exploration)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Learn Call', fontsize=11)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Stats
    final_std_x = all_std['move_x'][-1] if all_std['move_x'] else 0
    final_std_y = all_std['move_y'][-1] if all_std['move_y'] else 0
    final_std_r = all_std['rotation'][-1] if all_std['rotation'] else 0
    ax3.text(0.02, 0.98, f'Final X: {final_std_x:.3f}\nFinal Y: {final_std_y:.3f}\nFinal Rot: {final_std_r:.3f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5), fontsize=9)
    
    # 4. Critic Loss - LEARN CALLS en abscisse
    ax4 = axes[1, 1]
    ax4.plot(learn_calls, all_critic_loss, 'purple', alpha=0.6, linewidth=0.8)
    ax4.set_title('Critic Loss par Learn Call', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Learn Call', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Stats
    mean_loss = np.mean(all_critic_loss)
    final_loss = all_critic_loss[-1] if all_critic_loss else 0
    ax4.text(0.02, 0.98, f'Mean: {mean_loss:.4f}\nFinal: {final_loss:.4f}',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5), fontsize=9)
    
    # 5. Mean State Values (Critic Predictions) - LEARN CALLS
    ax5 = axes[1, 2]
    # Handle backward compatibility: pad with NaN if all_values is shorter
    if len(all_values) > 0 and len(all_values) < len(learn_calls):
        # Plot only the available values
        values_calls = learn_calls[:len(all_values)]
        ax5.plot(values_calls, all_values, 'teal', alpha=0.6, linewidth=0.8, label='State Values')
        ax5.axvline(x=len(all_values), color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Tracking started')
    elif len(all_values) > 0:
        ax5.plot(learn_calls, all_values, 'teal', alpha=0.6, linewidth=0.8)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax5.set_title('Mean State Value (Critic) par Learn Call', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Learn Call', fontsize=11)
    ax5.set_ylabel('State Value', fontsize=11)
    ax5.grid(True, alpha=0.3)
    if len(all_values) < len(learn_calls):
        ax5.legend(loc='best', fontsize=9)
    
    # Stats
    if len(all_values) > 0:
        mean_val = np.mean(all_values)
        final_val = all_values[-1]
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        ax5.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nFinal: {final_val:.2f}\nRange: [{min_val:.2f}, {max_val:.2f}]',
                 transform=ax5.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5), fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No data yet\n(metric added recently)',
                 transform=ax5.transAxes, ha='center', va='center',
                 fontsize=11, style='italic', color='gray')
    
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'BILAN_FINAL_training.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("📊 BILAN FINAL DE L'ENTRAÎNEMENT 📊")
    print("="*70)
    print(f"Total épisodes: {n_episodes}")
    print(f"\nReward - Mean: {mean_reward:.2f} | Max: {max_reward:.2f} | Min: {min_reward:.2f}")
    print(f"Entropy - Mean: {mean_entropy:.4f} | Final: {final_entropy:.4f}")
    print(f"Std - Final X: {final_std_x:.4f} | Y: {final_std_y:.4f} | Rot: {final_std_r:.4f}")
    print(f"Critic Loss - Mean: {mean_loss:.4f} | Final: {final_loss:.4f}")
    if len(all_values) > 0:
        mean_val = np.mean(all_values)
        final_val = all_values[-1]
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        print(f"State Values - Mean: {mean_val:.2f} | Final: {final_val:.2f} | Range: [{min_val:.2f}, {max_val:.2f}]")
    else:
        print(f"State Values - (not tracked in resumed checkpoint)")
    print(f"\n✅ Graphique sauvegardé: {filename}")
    print("="*70)


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

# alpha=0.0003
def train(n_games=1000, N=256, batch_size=64, n_epochs=15, alpha=0.0003,
          render=False, save_best=True, live_plot=True, plot_first_episode=True,
          resume=False, model_path='models/ppo', gamma=0.98, static_spawn=False):
    """
    Entraîne l'agent PPO sur Ping-Pong.
    
    Args:
        n_games: Nombre d'épisodes d'entraînement (parties complètes)
        N: Nombre de steps avant chaque mise à jour (512 = ~2-3 points)
        batch_size: Taille des mini-batches
        n_epochs: Nombre de fois qu'on réutilise les mêmes données par update
        alpha: Learning rate
        gamma: Discount factor (influence des rewards futurs)
        render: Afficher le jeu pendant l'entraînement
        save_best: Sauvegarder le meilleur modèle
        live_plot: Afficher un graphique en temps réel
        plot_first_episode: Sauvegarder le plot des rewards du premier épisode
        resume: Reprendre l'entraînement depuis le dernier modèle sauvegardé
        model_path: Chemin vers le modèle à charger/sauvegarder
    """
    # Créer l'environnement
    render_mode = "human" if render else None
    env = PingPongEnv(render_mode=render_mode, static_spawn=static_spawn)
    
    # Initialiser Pygame et un clock si render est activé (pour gestion events + tick)
    clock = None
    if render and pygame:
        pygame.init()
        clock = pygame.time.Clock()
    
    # Créer l'agent
    # Observation: embeddings (3x16) + 10 features continues = 58 total
    agent = Agent(
        n_actions=3,          # move_x, move_y, rotate
        input_dims=env.observation_space.shape[0],        # nombre de features continues (3 embeddings gérés séparément: ball, paddle, angle)
        gamma=gamma,          # Paramètre configurable (0.98 par défaut)
        alpha=alpha,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=batch_size,
        n_epochs=n_epochs,
        chkpt_dir=model_path,
        debug_adv=True
    )
    
    # Charger le modèle existant si resume=True
    if resume:
        actor_path = os.path.join(model_path, 'actor_torch_ppo')
        if os.path.exists(actor_path):
            agent.load_models()
            print(f"✅ Modèle chargé depuis {model_path}")
        else:
            print(f"⚠️ Aucun modèle trouvé dans {model_path}, démarrage from scratch")
    
    # === CHARGER L'ÉTAT D'ENTRAÎNEMENT COMPLET (si reprise) ===
    training_state = None
    start_episode = 0
    if resume:
        training_state = load_training_state(model_path)
        if training_state is not None:
            start_episode = training_state['episode'] + 1
            all_rewards = training_state['all_rewards']
            all_entropy = training_state['all_entropy']
            all_std = training_state['all_std']
            all_critic_loss = training_state['all_critic_loss']
            all_values = training_state.get('all_values', [])  # Backward compat
            score_history = training_state['score_history']
            best_score = training_state['best_score']
            print(f"🔄 État d'entraînement restauré")
        else:
            print(f"ℹ️  Démarrage d'un nouvel entraînement")
    
    figure_file = 'plots/pingpong_learning.png'
    
    best_score = float('-inf') if training_state is None else training_state['best_score']
    score_history = [] if training_state is None else training_state['score_history']
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    # Compteur du nombre d'appels à learn() pour les graphes
    learn_call_count = 0 if training_state is None else training_state.get('learn_call_count', 0)
    
    # Métriques GLOBALES (déjà restaurées ci-dessus si resume=True)
    if training_state is None:
        all_rewards = []
        all_entropy = []
        all_std = {'move_x': [], 'move_y': [], 'rotation': []}
        all_critic_loss = []
        all_values = []
    
    # Métriques pour tracking détaillé par siècle (100 épisodes)
    century_rewards = []  # Rewards par épisode pour le siècle en cours
    century_entropy = []  # Entropy par learn
    century_std = {'move_x': [], 'move_y': [], 'rotation': []}  # Std par learn
    century_critic_loss = []  # Critic loss par learn
    
    # Plus de moyennes par épisode pour entropy/std/loss; métriques stockées par learn
    
    # Tracking par phase pour afficher les stats
    phase_stats = {0: {'wins': 0, 'losses': 0}, 
                   1: {'wins': 0, 'losses': 0}, 
                   2: {'wins': 0, 'losses': 0}, 
                   3: {'wins': 0, 'losses': 0}}
    # Gamma progressif par phase (plus la phase est avancée, plus on regarde loin)
    # Calculs: γ^300 doit rester significatif pour relier cause et effet
    # Phase 0 (~80 steps): γ=0.995 → γ^80=0.67, γ^100=0.61
    # Phase 1 (~150 steps): γ=0.996 → γ^150=0.74, γ^200=0.67
    # Phase 2 (~350 steps): γ=0.998 → γ^300=0.74, γ^350=0.70
    # Phase 3 (long): γ=0.9985 → γ^400=0.82, γ^500=0.78
    phase_gamma = {0: 0.995, 1: 0.997, 2: 0.998, 3: 0.9985} # 0: 0.995, 1: 0.996, 2: 0.998, 3: 0.9985
    last_phase = -1
    
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
    if training_state is not None:
        print(f"Reprise à partir de l'épisode {start_episode + 1}/{n_games + start_episode}")
    print("=" * 50)

    reward_affichage = []

    for i in range(start_episode, start_episode + n_games):
        # === CURRICULUM LEARNING: Mettre à jour la phase ===
        env.set_episode_count(i)
        current_phase = env.training_phase
        # Adapter gamma selon la phase (plus on progresse, plus on anticipe loin)
        if current_phase in phase_gamma:
            agent.gamma = phase_gamma[current_phase]
        
        observation, _ = env.reset()
        done = False
        score = 0
        episode_rewards = []  # Track rewards de cet épisode
        episode_hits = 0
        
        last_info = None
        
        # Afficher la transition de phase (sauter si aucune phase précédente)
        if i > 0 and last_phase >= 0 and current_phase != last_phase:
            phase_names = ["Phase 0: Balle lente proche", "Phase 1: Balle modérée", 
                          "Phase 2: Balle normale", "Phase 3: Adversaire compétent"]
            
            # Afficher les stats de la phase précédente
            prev_phase = last_phase
            wins = phase_stats[prev_phase]['wins']
            losses = phase_stats[prev_phase]['losses']
            total = wins + losses
            win_pct = (wins / total * 100) if total > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"📊 FIN {phase_names[prev_phase]}")
            print(f"{'='*60}")
            print(f"Wins: {wins} | Losses: {losses} | Total: {total} | Win Rate: {win_pct:.1f}%")
            print(f"Gamma utilisé: {phase_gamma.get(prev_phase, gamma):.3f}")
            print(f"{'='*60}\n")
            
            # Reset stats pour la nouvelle phase
            phase_stats[current_phase] = {'wins': 0, 'losses': 0}
            
            print(f"🎓 TRANSITION: {phase_names[current_phase]} (Episode {i}) | Gamma={phase_gamma.get(current_phase, gamma):.3f}\n")
        
        last_phase = current_phase

        while not done:
            # Gérer les événements pygame uniquement si render est activé ET pygame existe
            if render and pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                if done:
                    break

            # Choisir une action
            action, prob, val = agent.choose_action(observation)

            # Exécuter l'action
            observation_, terminated, info = env.step(action)
            last_info = info  # garder le dernier info pour log de fin

            # Calculer la récompense pour l'entraînement
            reward = env.compute_reward(info)
            done = terminated
            
            reward_affichage.append(reward)

            n_steps += 1
            score += reward
            episode_rewards.append(reward)

            # Récupérer le nombre de hits depuis l'environnement
            episode_hits = info.get('agent_hits', 0)

            # Stocker la transition
            agent.remember(observation, action, prob, val, reward, done)

            # Apprendre tous les N steps
            if n_steps % N == 0:
                metrics = agent.learn()
                learn_iters += 1
                learn_call_count += 1
                # Accumuler directement par learn (global + siècle)
                all_entropy.append(metrics['entropy'])
                all_std['move_x'].append(metrics['std_move_x'])
                all_std['move_y'].append(metrics['std_move_y'])
                all_std['rotation'].append(metrics['std_rotation'])
                all_critic_loss.append(metrics['critic_loss'])
                all_values.append(metrics['mean_value'])

                century_entropy.append(metrics['entropy'])
                century_std['move_x'].append(metrics['std_move_x'])
                century_std['move_y'].append(metrics['std_move_y'])
                century_std['rotation'].append(metrics['std_rotation'])
                century_critic_loss.append(metrics['critic_loss'])

            observation = observation_

            # Cadencer le temps réel seulement si on affiche
            if render and clock:
                clock.tick(FPS)
        
        # Sauvegarder dans les listes GLOBALES (moyenne des rewards par step, pas somme)
        avg_reward_per_step = score / len(episode_rewards) if len(episode_rewards) > 0 else 0
        all_rewards.append(avg_reward_per_step)
        
        # Sauvegarder aussi dans les listes par SIÈCLE (moyenne par step)
        century_rewards.append(avg_reward_per_step)
        
        # Pas de moyennes par épisode pour entropy/std/loss (stockés par learn)
        
        # Plot tous les 100 épisodes (un siècle)
        if (i + 1) % 100 == 0 and len(century_rewards) > 0:
            start_ep = i + 1 - len(century_rewards) + 1
            end_ep = i + 1
            plot_century_metrics(
                episode_range=(start_ep, end_ep),
                rewards_history=century_rewards,
                entropy_history=century_entropy,
                std_history=century_std,
                critic_loss_history=century_critic_loss
            )
            # Réinitialiser les listes pour le prochain siècle
            century_rewards = []
            century_entropy = []
            century_std = {'move_x': [], 'move_y': [], 'rotation': []}
            century_critic_loss = []
        
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
            phase_stats[current_phase]['wins'] += 1
        elif episode_winner == 'right' and env.agent_side == 'right':
            won = "✓"
            phase_stats[current_phase]['wins'] += 1
        else:
            won = "✗"
            phase_stats[current_phase]['losses'] += 1

        # Afficher la progression
        phase_names = {0: "🟦", 1: "🟩", 2: "🟨", 3: "🟥"}  # Couleurs pour phases
        phase_indicator = phase_names.get(current_phase, "⚪")
        print(f'Ep {i+1:4d} [{phase_indicator} P{current_phase}] | Score: {score:7.1f} | Avg: {avg_score:7.1f} | '
              f'Hits: {episode_hits} | Won: {won} | Steps: {len(episode_rewards):3d}')

        # Log détaillé de fin d'épisode (faute / vainqueur) pour debug même sans render
        if last_info is not None:
            faults = last_info.get('faults', {})
            winner = last_info.get('winner_side')
            print(f"    EndReason | winner_side={winner} | faults={faults}")
        
        # Mettre à jour le plot live
        if live_plot and fig is not None:
            update_live_plot(fig, ax1, ax2, score_history, update_freq=10)
        
        # === SAUVEGARDER L'ÉTAT D'ENTRAÎNEMENT régulièrement ===
        # Sauvegarder tous les 10 épisodes (ou à la fin)
        if (i + 1 - start_episode) % 10 == 0 or (i + 1 - start_episode) == n_games:
            save_training_state(
                model_path,
                episode=i,  # Numéro d'épisode absolu
                all_rewards=all_rewards,
                all_entropy=all_entropy,
                all_std=all_std,
                all_critic_loss=all_critic_loss,
                all_values=all_values,
                score_history=score_history,
                best_score=best_score,
                training_phase=current_phase,  # Sauvegarder la phase
                learn_call_count=learn_call_count
            )
    
    print(f"reward affichage : {reward_affichage}")

    # Fermer le plot interactif
    if live_plot:
        plt.ioff()
        plt.close('all')
    
    # Tracer la courbe d'apprentissage finale (ancienne version avec moyenne glissante)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
    # Tracer le BILAN FINAL avec toutes les métriques (valeurs réelles)
    plot_final_summary(all_rewards, all_entropy, all_std, all_critic_loss, all_values)
    
    env.close()
    print("\n=== Entraînement terminé ===")
    
    return agent, score_history


def _update_ball_debug_info(game):
    """Met à jour les infos de debug pour la vitesse et le spin de la balle."""
    if game.env.ball_in_play and game.env.ball:
        game.last_ball_vel = (game.env.ball.vel[0], game.env.ball.vel[1])
        game.last_spin = game.env.ball.angular_speed


def play_ai_vs_ai(model_path='models/ppo', num_episodes=5, vs_trained=False):
    """
    IA vs IA avec affichage visuel (évaluation sans entraînement).
    
    Args:
        model_path: Chemin vers les modèles sauvegardés
        num_episodes: Nombre d'épisodes à jouer
        vs_trained: Si False (défaut), IA entraînée vs IA simple
                   Si True, IA entraînée vs IA entraînée (deux instances du même modèle)
    """
    # Vérifier que le modèle existe
    actor_path = os.path.join(model_path, 'actor_torch_ppo')
    critic_path = os.path.join(model_path, 'critic_torch_ppo')
    
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"❌ Erreur: Modèles incomplets dans {model_path}")
        print(f"   Actor: {'✅' if os.path.exists(actor_path) else '❌'}")
        print(f"   Critic: {'✅' if os.path.exists(critic_path) else '❌'}")
        print("   Lance d'abord l'entraînement avec: python train.py")
        return
    
    # Importer Game uniquement quand nécessaire
    from engine.game import Game
    
    agent_left = Agent(
        n_actions=3,
        input_dims=14,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent_left.load_models()
    print(f"✅ Modèle gauche chargé depuis {model_path}")
    
    # Créer l'agent droite si vs_trained
    agent_right = None
    if vs_trained:
        agent_right = Agent(
            n_actions=3,
            input_dims=10,
            gamma=0.99,
            alpha=0.0003,
            chkpt_dir=model_path
        )
        agent_right.load_models()
        print(f"✅ Modèle droite chargé depuis {model_path}")
    
    print("=== Mode Jeu IA vs IA ===")
    if vs_trained:
        print("IA gauche: Modèle entraîné")
        print("IA droite: Modèle entraîné")
    else:
        print("IA gauche: Modèle entraîné")
        print("IA droite: IA simple basique")
    
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
            action_p1 = predict_action(agent_left, obs, deterministic=True)
            
            # IA droite (opponent_paddle)
            if vs_trained and agent_right is not None:
                # Utiliser le modèle entraîné
                action_p2 = predict_action(agent_right, obs, deterministic=True)
            else:
                # Utiliser l'IA simple intégrée
                action_p2 = game.env._get_opponent_action()
            
            # Simuler directement via env (pas de reward car pas d'entraînement)
            obs, done, info = game.env.step(action_p1, action_p2)
            game.score_left = info.get('score_left', 0)
            game.score_right = info.get('score_right', 0)
            game.point_message = info.get('point_message', '')
            
            steps += 1
            
            # Mettre à jour les infos de debug (vitesse et spin de la balle)
            _update_ball_debug_info(game)
            
            # Afficher visuellement
            game.draw()
            game.clock.tick(FPS)
        
        if not game.running:
            break
        
        print(f"Episode {episode + 1}: {game.score_left} - {game.score_right}")
    
    game.running = False
    pygame.quit()


def play_human_vs_human(mouse_control_p1=False):
    """
    1v1 entre deux joueurs humains avec affichage complet.
    Joueur 1 (gauche): Souris si mouse_control_p1=True, sinon Z/S (vertical), Q/D (horizontal), A/E (rotation)
    Joueur 2 (droite): O/L (vertical), K/M (horizontal), I/P (rotation)
    """
    from engine.game import Game
    
    print("=== Mode Humain vs Humain ===")
    if mouse_control_p1:
        print("Joueur 1 (gauche): SOURIS (clics pour rotation)")
    else:
        print("Joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
    print("Joueur 2 (droite): O/L=vertical, K/M=horizontal, I/P=rotation")
    
    game = Game(player1_type="human", player2_type="human", mouse_control_p1=mouse_control_p1)
    game.run()
    
    print("Fin du jeu!")


def play_ai_vs_human(model_path='models/ppo', mouse_control_p1=False):
    """
    IA vs Joueur Humain avec affichage complet.
    
    Args:
        model_path: Chemin vers les modèles sauvegardés
        mouse_control_p1: Si True, contrôler joueur 1 à la souris
    """
    # Vérifier que le modèle existe
    actor_path = os.path.join(model_path, 'actor_torch_ppo')
    critic_path = os.path.join(model_path, 'critic_torch_ppo')
    
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print(f"❌ Erreur: Modèles incomplets dans {model_path}")
        print(f"   Actor: {'✅' if os.path.exists(actor_path) else '❌'}")
        print(f"   Critic: {'✅' if os.path.exists(critic_path) else '❌'}")
        print("   Lance d'abord l'entraînement avec: python train.py")
        return
    
    from engine.game import Game
    
    agent = Agent(
        n_actions=3,
        input_dims=14,
        gamma=0.99,
        alpha=0.0003,
        chkpt_dir=model_path
    )
    agent.load_models()
    print(f"✅ Modèles chargés depuis {model_path}")
    
    print("=== Mode IA vs Humain ===")
    if mouse_control_p1:
        print("Vous êtes le joueur 1 (gauche): SOURIS (clics pour rotation)")
    else:
        print("Vous êtes le joueur 1 (gauche): Z/S=vertical, Q/D=horizontal, A/E=rotation")
    print("L'IA est le joueur 2 (droite)")
    
    game = Game(player1_type="human", player2_type="ai", mouse_control_p1=mouse_control_p1)
    
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
        
        # Mettre à jour les infos de debug (vitesse et spin de la balle)
        _update_ball_debug_info(game)
        
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
    parser.add_argument('--mouse', action='store_true',
                        help='Activer le contrôle à la souris pour le joueur 1 (modes: human, ai_vs_human)')
    parser.add_argument('--static_spawn', action='store_true',
                        help='Spawn la balle immobile sans gravité jusqu’au premier contact raquette')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Par défaut resume=True, sauf si --fresh est spécifié
        should_resume = args.resume and not args.fresh
        train(n_games=args.episodes, render=args.render, live_plot=args.render_plot,
              resume=should_resume, model_path=args.model_path, static_spawn=args.static_spawn)
    elif args.mode == 'play':
        play_ai_vs_ai(model_path=args.model_path, num_episodes=args.episodes)
    elif args.mode == 'human':
        play_human_vs_human(mouse_control_p1=args.mouse)
    elif args.mode == 'ai_vs_human':
        play_ai_vs_human(model_path=args.model_path, mouse_control_p1=args.mouse)

# Mode 1: Entraînement basique
# # Basique (1000 épisodes, reprend depuis modèle existant)
# python train.py --mode train

# # Avec affichage du jeu
# python train.py --mode train --render

# # Avec graphique en temps réel
# python train.py --mode train --render_plot

# # Personnaliser le nombre d'épisodes
# python train.py --mode train --episodes 500 --resume
# python train.py --mode train --episodes 2000 --resume

# # Combiner options
# python train.py --mode train --episodes 10 --render --resume

# # Démarrer un nouvel entraînement (oublier ancien modèle)
# python train.py --mode train --fresh
# python train.py --mode train --fresh --episodes 500 --render_plot

# # Avec chemin modèle personnalisé
# python train.py --mode train --model_path models/custom_model



# Mode 2: IA vs IA affiché
# # Basique (5 parties)
# python train.py --mode play

# # Nombre de parties
# python train.py --mode play --episodes 10
# python train.py --mode play --episodes 3

# # Avec modèle personnalisé
# python train.py --mode play --model_path models/custom_model
# python train.py --mode play --episodes 20 --model_path models/custom_model



# Mode 3: Humain vs Humain
# # Clavier (défaut - ZQSD + AE pour joueur 1)
# python train.py --mode human

# # Souris pour joueur 1 (clics pour rotation)
# python train.py --mode human --mouse



# Mode 4: IA vs Humain
# # Clavier (défaut - ZQSD + AE)
# python train.py --mode ai_vs_human

# # Souris pour vous (le joueur 1)
# python train.py --mode ai_vs_human --mouse

# # Avec modèle personnalisé
# python train.py --mode ai_vs_human --model_path models/custom_model
# python train.py --mode ai_vs_human --model_path models/custom_model --mouse