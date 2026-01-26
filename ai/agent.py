"""
Agent PPO pour le Ping-Pong.
Implémentation from scratch avec PyTorch (style Phil's code).
Adapté pour actions continues.
"""

import os
import numpy as np
import torch as T
from torch.distributions import Normal

from ai.model import ActorNetwork, CriticNetwork
from ai.memory import PPOMemory


class Agent:
    """
    Agent PPO pour actions continues.
    """
    def __init__(self, n_actions, input_dims, gamma=0.997, alpha=0.0003, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 chkpt_dir='models/ppo', debug_adv=False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)
        # Debug des avantages (GAE) pour comprendre le signal d'apprentissage
        self.debug_adv = debug_adv
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """
        Choisit une action à partir de l'observation.
        
        Args:
            observation: dict avec 'ball_idx', 'paddle_idx', 'angle_idx', 'continuous' ou array simple
        
        Returns:
            action: np.array de shape (n_actions,)
            log_prob: log probabilité corrigée (tanh-squashed) de l'action
            value: valeur estimée de l'état
        """
        # Préparer l'état pour le réseau
        if isinstance(observation, dict):
            # Format avec embeddings
            state = {
                'ball_idx': T.tensor([observation['ball_idx']], dtype=T.long).to(self.actor.device),
                'paddle_idx': T.tensor([observation['paddle_idx']], dtype=T.long).to(self.actor.device),
                'angle_idx': T.tensor([observation['angle_idx']], dtype=T.long).to(self.actor.device),
                'continuous': T.tensor([observation['continuous']], dtype=T.float).to(self.actor.device)
            }
        else:
            # Format classique (array)
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        print(f"Debug: Choosing action : {state.shape = }")

        with T.no_grad():
            mu, std = self.actor(state)
            value = self.critic(state)
        
        # Squashed Gaussian: échantillonner u ~ N(mu, std), action a = tanh(u)
        dist = Normal(mu, std)
        u = dist.sample()
        a = T.tanh(u)
        
        # Log-probabilité corrigée via changement de variables
        # log pi(a) = log N(u) - sum log(1 - tanh(u)^2 + eps) ; ici tanh(u)=a
        eps = 1e-6
        log_prob = dist.log_prob(u).sum(dim=-1) - T.sum(T.log(1 - a.pow(2) + eps), dim=-1)

        action = a.squeeze().cpu().detach().numpy()
        log_prob = log_prob.squeeze().item()
        value = value.squeeze().item()

        return action, log_prob, value

    def learn(self):
        # Métriques à retourner
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_std = np.zeros(self.n_actions)
        total_critic_values = 0.0  # Track mean state values
        num_batches = 0
        
        # Flag pour debug des high actor loss (une seule fois par learn)
        high_loss_logged = False
        
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calcul GAE (Optimisé O(N) - Backwards)
            last_gae_lam = 0
            # On itère à l'envers, de la fin vers le début
            for t in reversed(range(len(reward_arr) - 1)):
                # Delta = erreur de prédiction TD (Temporal Difference)
                delta = reward_arr[t] + self.gamma * values[t+1] * (1 - int(dones_arr[t])) - values[t]
                
                # Formule récursive du GAE
                last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * last_gae_lam
                advantage[t] = last_gae_lam
            
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # Normalisation des avantages (Crucial pour la stabilité)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for batch in batches:
                # Traiter les états (peuvent être des dicts avec embeddings)
                batch_states = state_arr[batch]
                if isinstance(batch_states[0], dict):
                    # Format avec embeddings spatiaux
                    ball_indices = T.tensor([s['ball_idx'] for s in batch_states], dtype=T.long).to(self.actor.device)
                    paddle_indices = T.tensor([s['paddle_idx'] for s in batch_states], dtype=T.long).to(self.actor.device)
                    angle_indices = T.tensor([s['angle_idx'] for s in batch_states], dtype=T.long).to(self.actor.device)
                    continuous_features = T.tensor([s['continuous'] for s in batch_states], dtype=T.float).to(self.actor.device)
                    states = {
                        'ball_idx': ball_indices,
                        'paddle_idx': paddle_indices,
                        'angle_idx': angle_indices,
                        'continuous': continuous_features
                    }
                else:
                    # Format classique (array simple)
                    states = T.tensor(batch_states, dtype=T.float).to(self.actor.device)
                
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

                # Forward pass
                mu, std = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value, dim=-1)

                # Calculer les nouvelles log probs (tanh-squashed)
                dist = Normal(mu, std)
                eps = 1e-6
                # actions sont déjà dans [-1,1]; remonter u = atanh(a)
                a = actions.clamp(-1 + eps, 1 - eps)
                u = T.atanh(a)  # Numériquement plus stable
                new_probs = dist.log_prob(u).sum(dim=-1) - T.sum(T.log(1 - a.pow(2) + eps), dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()  # mesure du chaos
                # entropy élevé -> courbe plate -> exploration
                # entropy faible -> courbe pointue -> exploitation
                
                # Ratio pour PPO
                prob_ratio = (new_probs - old_probs).exp() # new_probs / old_probs en log space
                
                # Loss acteur (PPO clipped)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 
                                                  1 - self.policy_clip,
                                                  1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Loss critique avec Clipping (PPO standard pour stabilité)
                returns = advantage[batch] + values[batch]
                v_pred = critic_value
                v_old = values[batch]
                
                # 1. Loss standard
                v_loss1 = (returns - v_pred) ** 2
                
                # 2. Loss clippée - force la valeur à rester proche de l'ancienne
                v_pred_clipped = v_old + T.clamp(v_pred - v_old, -self.policy_clip, self.policy_clip)
                v_loss2 = (returns - v_pred_clipped) ** 2
                
                # On prend le max des deux (contrainte la plus dure)
                critic_loss = 0.5 * T.max(v_loss1, v_loss2).mean()
                # Cas 1 – État réellement bon (gagner le point) mais sous-estimé
                # returns ≫ critic_value → erreur élevée → loss augmente → le critique apprend à monter sa prédiction.
                # Cas 2 – État réellement mauvais (perdre le point) mais surestimé
                # returns ≪ critic_value → erreur élevée → loss augmente → le critique apprend à baisser sa prédiction.
                # Cas 3 – État neutre / transitions courtes (peu de reward)
                # returns ≈ critic_value → erreur faible → loss faible → ajustements minimes.
                # Cas 4 – Rewards rares et forts (ping-pong)
                # Quand un point est gagné/perdu, returns fait un saut (±100) → si le critique ne l’avait pas anticipé, la loss grimpe, forçant une mise à jour importante pour mieux prévoir ces transitions.

                
                # Loss totale avec entropy bonus (c2 réduit pour éviter d'entretenir une std élevée
                # sur des dimensions à faible avantage comme move_x/move_y)
                total_loss = actor_loss + critic_loss - 0.001 * entropy

                #print(f"Debug Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
                
                # Backpropagation
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # Clipping des gradients pour éviter l'explosion (PPO: max_norm=1.0)
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                # Accumuler les métriques
                actor_loss_value = actor_loss.item()
                total_actor_loss += actor_loss_value
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_std += std.mean(dim=0).detach().cpu().numpy()
                total_critic_values += critic_value.mean().item()  # Track actual values
                num_batches += 1
                
                # 🔍 DEBUG: Actor loss élevée (>10)
                if not high_loss_logged and actor_loss_value > 10.0:
                    high_loss_logged = True
                    print(f"\n⚠️  HIGH ACTOR LOSS DETECTED: {actor_loss_value:.4f}")
                    print(f"    Advantage batch stats:")
                    print(f"      - Mean: {advantage[batch].mean().item():.4f}")
                    print(f"      - Std: {advantage[batch].std().item():.4f}")
                    print(f"      - Min: {advantage[batch].min().item():.4f}")
                    print(f"      - Max: {advantage[batch].max().item():.4f}")
                    print(f"    Prob ratio stats:")
                    print(f"      - Mean: {prob_ratio.mean().item():.4f}")
                    print(f"      - Min: {prob_ratio.min().item():.4f}")
                    print(f"      - Max: {prob_ratio.max().item():.4f}")
                    print(f"    Weighted probs: {weighted_probs.mean().item():.4f}")
                    print(f"    Clipped probs: {weighted_clipped_probs.mean().item():.4f}")
                    print(f"    Entropy: {entropy.item():.4f}")
                    print(f"    Std (mean): {std.mean().item():.4f}\n")

        self.memory.clear_memory()
        
        # Retourner les métriques moyennes
        metrics = {
            'actor_loss': total_actor_loss / max(num_batches, 1),
            'critic_loss': total_critic_loss / max(num_batches, 1),
            'entropy': total_entropy / max(num_batches, 1),
            'std_move_x': total_std[0] / max(num_batches, 1),
            'std_move_y': total_std[1] / max(num_batches, 1),
            'std_rotation': total_std[2] / max(num_batches, 1),
            'mean_value': total_critic_values / max(num_batches, 1)  # Average state value
        }
        return metrics


def predict_action(agent, observation, deterministic=False):
    """
    Prédit une action pour le jeu (sans exploration si deterministic).
    
    Args:
        observation: dict avec 'ball_idx', 'paddle_idx', 'continuous' ou array simple
    """
    # Préparer l'état
    if isinstance(observation, dict):
        state = {
            'ball_idx': T.tensor([observation['ball_idx']], dtype=T.long).to(agent.actor.device),
            'paddle_idx': T.tensor([observation['paddle_idx']], dtype=T.long).to(agent.actor.device),
            'angle_idx': T.tensor([observation['angle_idx']], dtype=T.long).to(agent.actor.device),
            'continuous': T.tensor([observation['continuous']], dtype=T.float).to(agent.actor.device)
        }
    else:
        state = T.tensor([observation], dtype=T.float).to(agent.actor.device)
    
    mu, std = agent.actor(state)
    
    if deterministic:
        action = T.tanh(mu)
    else:
        dist = Normal(mu, std)
        u = dist.sample()
        action = T.tanh(u)
    
    # Convertir vers numpy de manière robuste
    action_np = action.squeeze().detach()
    if action_np.is_cuda:
        action_np = action_np.cpu()
    return action_np.numpy()
