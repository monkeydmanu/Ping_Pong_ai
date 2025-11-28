"""
Configuration du réseau de neurones pour PPO.
Architectures personnalisées si besoin.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class PingPongFeatureExtractor(BaseFeaturesExtractor):
    """
    Extracteur de features personnalisé pour le Ping-Pong.
    
    Architecture:
        Input (12) -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> Output
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


# Configuration des policy_kwargs pour utiliser l'extracteur personnalisé
CUSTOM_POLICY_KWARGS = {
    "features_extractor_class": PingPongFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 128},
    "net_arch": {
        "pi": [64, 64],  # Réseau de la politique (acteur)
        "vf": [64, 64]   # Réseau de valeur (critique)
    }
}

# Configuration standard (plus simple, souvent suffisante)
STANDARD_POLICY_KWARGS = {
    "net_arch": [256, 256]  # 2 couches cachées de 256 neurones
}


def get_policy_kwargs(use_custom=False):
    """
    Retourne les kwargs de politique à utiliser.
    
    Args:
        use_custom: Si True, utilise l'architecture personnalisée
    
    Returns:
        dict: Arguments pour la politique PPO
    """
    if use_custom:
        return CUSTOM_POLICY_KWARGS
    return STANDARD_POLICY_KWARGS
