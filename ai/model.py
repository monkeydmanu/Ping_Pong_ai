"""
Réseaux de neurones pour PPO (Acteur-Critique).
Adapté pour actions continues avec distribution Gaussienne.
"""

import os
import torch as T
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    """
    Réseau Acteur pour actions continues.
    Produit la moyenne (mu) des actions, sigma est appris séparément.
    """
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='models/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        
        # Réseau pour la moyenne des actions
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh()  # Sortie entre [-1, 1] pour actions continues
        )
        
        # Log de l'écart-type (appris)
        # Initialisé à 0 -> sigma = exp(0) = 1
        self.log_std = nn.Parameter(T.zeros(n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Retourne la moyenne et l'écart-type de la distribution.
        """
        mu = self.actor(state)
        # Clamp log_std pour éviter des valeurs extrêmes
        log_std = T.clamp(self.log_std, -20, 2)
        std = log_std.exp().expand_as(mu)
        
        return mu, std

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    """
    Réseau Critique - estime la valeur V(s).
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='models/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
