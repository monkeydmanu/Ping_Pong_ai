"""
Agent PPO pour le Ping-Pong.
Utilise Stable-Baselines3 pour l'entraînement.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ai.environment import PingPongEnv


class PingPongAgent:
    """
    Agent PPO pour jouer au Ping-Pong.
    """
    
    def __init__(self, model_path=None):
        """
        Initialise l'agent.
        
        Args:
            model_path: Chemin vers un modèle pré-entraîné (optionnel)
        """
        self.model = None
        self.env = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def create_model(self, env, **kwargs):
        """
        Crée un nouveau modèle PPO.
        
        Args:
            env: Environnement Gymnasium
            **kwargs: Arguments supplémentaires pour PPO
        """
        default_params = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,           # Facteur de discount
            "gae_lambda": 0.95,      # GAE lambda
            "clip_range": 0.2,       # Clipping PPO
            "ent_coef": 0.01,        # Coefficient d'entropie (exploration)
            "verbose": 1,
            "tensorboard_log": "./tensorboard_logs/"
        }
        
        # Fusionner avec les paramètres personnalisés
        default_params.update(kwargs)
        
        self.model = PPO(env=env, **default_params)
        self.env = env
        
        return self.model
    
    def train(self, total_timesteps=100000, save_path="models/ppo_pingpong", 
              save_freq=10000, eval_freq=5000):
        """
        Entraîne l'agent.
        
        Args:
            total_timesteps: Nombre total de steps d'entraînement
            save_path: Chemin pour sauvegarder les checkpoints
            save_freq: Fréquence de sauvegarde (en steps)
            eval_freq: Fréquence d'évaluation (en steps)
        """
        if self.model is None:
            raise ValueError("Modèle non initialisé. Appelez create_model() d'abord.")
        
        # Créer le dossier de sauvegarde
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Callback pour sauvegarder régulièrement
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.dirname(save_path),
            name_prefix="ppo_pingpong"
        )
        
        # Entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Sauvegarder le modèle final
        self.model.save(save_path)
        print(f"Modèle sauvegardé: {save_path}")
    
    def predict(self, observation, deterministic=True):
        """
        Prédit une action à partir d'une observation.
        
        Args:
            observation: Observation de l'environnement
            deterministic: Si True, utilise la politique déterministe
        
        Returns:
            action: Action à effectuer
        """
        if self.model is None:
            raise ValueError("Modèle non chargé.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def load(self, model_path):
        """
        Charge un modèle pré-entraîné.
        
        Args:
            model_path: Chemin vers le modèle
        """
        self.model = PPO.load(model_path)
        print(f"Modèle chargé: {model_path}")
    
    def save(self, model_path):
        """
        Sauvegarde le modèle actuel.
        
        Args:
            model_path: Chemin de sauvegarde
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Modèle sauvegardé: {model_path}")


def train_agent(total_timesteps=100000, n_envs=4, render=False):
    """
    Fonction utilitaire pour entraîner un agent.
    
    Args:
        total_timesteps: Nombre de steps d'entraînement
        n_envs: Nombre d'environnements parallèles
        render: Si True, affiche le jeu pendant l'entraînement
    """
    # Créer des environnements parallèles pour accélérer l'entraînement
    def make_env():
        return PingPongEnv(render_mode="human" if render else None)
    
    if n_envs > 1:
        env = make_vec_env(make_env, n_envs=n_envs)
    else:
        env = DummyVecEnv([make_env])
    
    # Créer et entraîner l'agent
    agent = PingPongAgent()
    agent.create_model(env)
    agent.train(total_timesteps=total_timesteps)
    
    env.close()
    return agent


def play_with_agent(model_path, num_episodes=5):
    """
    Joue avec un agent entraîné (pour visualiser).
    
    Args:
        model_path: Chemin vers le modèle
        num_episodes: Nombre d'épisodes à jouer
    """
    env = PingPongEnv(render_mode="human")
    agent = PingPongAgent(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
    
    env.close()


# Point d'entrée pour l'entraînement
if __name__ == "__main__":
    print("=== Entraînement de l'agent PPO ===")
    agent = train_agent(total_timesteps=50000, n_envs=1, render=True)
