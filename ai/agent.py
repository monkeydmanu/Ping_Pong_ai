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
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 chkpt_dir='models/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)
       
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
        
        Returns:
            action: np.array de shape (n_actions,)
            log_prob: log probabilité de l'action (somme sur toutes les dimensions)
            value: valeur estimée de l'état
        """
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        mu, std = self.actor(state)
        value = self.critic(state)
        
        # Distribution normale pour chaque action
        dist = Normal(mu, std)
        action = dist.sample()
        
        # Log prob = somme des log probs de chaque dimension
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Champ action entre [-1, 1]
        action = T.clamp(action, -1.0, 1.0)

        action = action.squeeze().cpu().detach().numpy()
        log_prob = log_prob.squeeze().item()
        value = value.squeeze().item()

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calcul GAE (Generalized Advantage Estimation)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] *\
                            (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

                # Forward pass
                mu, std = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                # Calculer les nouvelles log probs
                dist = Normal(mu, std)
                new_probs = dist.log_prob(actions).sum(dim=-1)
                
                # Ratio pour PPO
                prob_ratio = (new_probs - old_probs).exp()
                
                # Loss acteur (PPO clipped)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 
                                                  1 - self.policy_clip,
                                                  1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Loss critique (MSE)
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Loss totale
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Backpropagation
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


def predict_action(agent, observation, deterministic=False):
    """
    Prédit une action pour le jeu (sans exploration si deterministic).
    """
    state = T.tensor([observation], dtype=T.float).to(agent.actor.device)
    
    mu, std = agent.actor(state)
    
    if deterministic:
        action = mu
    else:
        dist = Normal(mu, std)
        action = dist.sample()
    
    action = T.clamp(action, -1.0, 1.0)
    return action.squeeze().cpu().detach().numpy()
