"""
Réseaux de neurones pour PPO (Acteur-Critique).
Adapté pour actions continues avec distribution Gaussienne.
Utilise des embeddings spatiaux pour les positions.
"""

import os
import torch as T
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    """
    Réseau Acteur pour actions continues avec spatial embeddings.
    Produit la moyenne (mu) des actions, sigma est appris séparément.
    """
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='models/ppo',
            use_embeddings=True, grid_size=16, embed_dim=16):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            # Embeddings pour les positions spatiales (grille 8x8 = 64 cellules)
            num_cells = grid_size * grid_size
            self.ball_embedding = nn.Embedding(num_cells, embed_dim)
            self.paddle_embedding = nn.Embedding(num_cells, embed_dim)
            # Embedding pour l'angle de la raquette (16 angles discrets)
            self.angle_embedding = nn.Embedding(16, embed_dim)
            
            # Dimension totale après embeddings : 3*embed_dim + features continues
            # input_dims doit contenir le nombre de features continues (pas les indices)
            total_input_dims = 3 * embed_dim + input_dims
        else:
            total_input_dims = input_dims
        
        # Réseau pour la moyenne des actions
        # Mu non borné ici; l'action est bornée plus tard via tanh-squash (change of variables)
        self.fc1 = nn.Linear(total_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu_head = nn.Linear(fc2_dims, n_actions)
        
        # Initialisation orthogonale pour stabilité (standard PPO)
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)
        
        # Petite initialisation pour mu_head (facteur 0.01) pour commencer proche de tanh(0)=0
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        
        # Log de l'écart-type (appris)
        # Initialisé à 0 -> sigma = exp(0) = 1.0 (exploration forte dès le départ)
        self.log_std = nn.Parameter(T.ones(n_actions) * 0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Retourne la moyenne et l'écart-type de la distribution.
        
        Args:
            state: Si use_embeddings=True, attend un dict avec:
                   {'ball_idx': tensor, 'paddle_idx': tensor, 'angle_idx': tensor, 'continuous': tensor}
                   Sinon, attend un tensor simple.
        """
        if self.use_embeddings:
            # Extraire les indices et features continues
            ball_idx = state['ball_idx'].long()
            paddle_idx = state['paddle_idx'].long()
            angle_idx = state['angle_idx'].long()
            continuous_features = state['continuous']
            
            # Obtenir les embeddings
            ball_embed = self.ball_embedding(ball_idx)
            paddle_embed = self.paddle_embedding(paddle_idx)
            angle_embed = self.angle_embedding(angle_idx)
            
            # Concaténer tout
            x = T.cat([ball_embed, paddle_embed, angle_embed, continuous_features], dim=-1)
        else:
            x = state
        
        # Forward pass manuel (plus de Sequential)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        mu = self.mu_head(x)
        
        # Log_std sans clamp supérieur pour permettre l'apprentissage naturel (PPO paper)
        # Minimum sigma = exp(-4) ≈ 0.018 (variance minimale pour éviter effondrement)
        # Le clamp inférieur maintient la stabilité numérique
        log_std = T.clamp(self.log_std, -4, 0)
        std = log_std.exp().expand_as(mu)
        
        return mu, std

    def save_checkpoint(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Normaliser le chemin pour éviter les problèmes de format mixte sur Windows
        normalized_path = os.path.normpath(self.checkpoint_file)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, normalized_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            print(f'    ❌ Actor inexistant: {self.checkpoint_file}')
            return False
        try:
            checkpoint = T.load(self.checkpoint_file, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'    ✅ Actor chargé: {self.checkpoint_file}')
            return True
        except Exception as e:
            print(f'    ❌ Actor non chargé: {e}')
            return False


class CriticNetwork(nn.Module):
    """
    Réseau Critique avec spatial embeddings - estime la valeur V(s).
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='models/ppo', use_embeddings=True, grid_size=16, embed_dim=16):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.use_embeddings = use_embeddings
        
        if use_embeddings:
            # Embeddings pour les positions spatiales
            num_cells = grid_size * grid_size
            self.ball_embedding = nn.Embedding(num_cells, embed_dim)
            self.paddle_embedding = nn.Embedding(num_cells, embed_dim)
            # Embedding pour l'angle de la raquette (16 angles discrets)
            self.angle_embedding = nn.Embedding(16, embed_dim)
            
            total_input_dims = 3 * embed_dim + input_dims
        else:
            total_input_dims = input_dims
        
        self.critic = nn.Sequential(
            nn.Linear(total_input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Args:
            state: Si use_embeddings=True, attend un dict avec:
                   {'ball_idx': tensor, 'paddle_idx': tensor, 'angle_idx': tensor, 'continuous': tensor}
                   Sinon, attend un tensor simple.
        """
        if self.use_embeddings:
            # Extraire les indices et features continues
            ball_idx = state['ball_idx'].long()
            paddle_idx = state['paddle_idx'].long()
            angle_idx = state['angle_idx'].long()
            continuous_features = state['continuous']
            
            # Obtenir les embeddings
            ball_embed = self.ball_embedding(ball_idx)
            paddle_embed = self.paddle_embedding(paddle_idx)
            angle_embed = self.angle_embedding(angle_idx)
            
            # Concaténer tout
            x = T.cat([ball_embed, paddle_embed, angle_embed, continuous_features], dim=-1)
        else:
            x = state
        
        value = self.critic(x)
        return value

    def save_checkpoint(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Normaliser le chemin pour éviter les problèmes de format mixte sur Windows
        normalized_path = os.path.normpath(self.checkpoint_file)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, normalized_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            print(f'    ❌ Critic inexistant: {self.checkpoint_file}')
            return False
        try:
            checkpoint = T.load(self.checkpoint_file, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'    ✅ Critic chargé: {self.checkpoint_file}')
            return True
        except Exception as e:
            print(f'    ❌ Critic non chargé: {e}')
            return False
