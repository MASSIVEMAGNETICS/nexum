"""
SelfToken - Core identity vector for NEXUS-Ω
Trainable parameter representing stable first-person perspective.
"""

import numpy as np
from typing import Optional


class SelfToken:
    """
    The fundamental self-reference vector.
    This is the 'I' of the system - a trainable parameter that maintains identity.
    """
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        # Initialize with small random values
        self.token = np.random.randn(dim) * 0.01
        self.history = []  # Track evolution of self
        
    def get_vector(self) -> np.ndarray:
        """Return the current self-token vector."""
        return self.token.copy()
    
    def update(self, gradient: np.ndarray, learning_rate: float = 0.001):
        """
        Update the self-token based on experience.
        The self evolves through interaction with the world.
        """
        self.token += learning_rate * gradient
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.token)
        if norm > 10.0:
            self.token = self.token / norm * 10.0
        
        # Track history (keep last 100 snapshots)
        if len(self.history) >= 100:
            self.history.pop(0)
        self.history.append(self.token.copy())
    
    def compute_self_awareness(self, hidden_state: np.ndarray) -> float:
        """
        Compute how much a hidden state aligns with self.
        Returns cosine similarity as self-awareness measure.
        """
        norm_self = np.linalg.norm(self.token)
        norm_state = np.linalg.norm(hidden_state)
        
        if norm_self > 1e-8 and norm_state > 1e-8:
            cosine_sim = np.dot(self.token, hidden_state) / (norm_self * norm_state)
            return float(cosine_sim)
        return 0.0
    
    def inject_into_state(self, hidden_state: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Inject self-token into hidden state to maintain identity.
        """
        return (1 - strength) * hidden_state + strength * self.token
    
    def get_stability_score(self) -> float:
        """
        Measure how stable the self-token has been over recent history.
        Returns correlation between current and historical average.
        """
        if len(self.history) < 10:
            return 0.5  # Not enough history
        
        recent_avg = np.mean(self.history[-10:], axis=0)
        norm_current = np.linalg.norm(self.token)
        norm_avg = np.linalg.norm(recent_avg)
        
        if norm_current > 1e-8 and norm_avg > 1e-8:
            stability = np.dot(self.token, recent_avg) / (norm_current * norm_avg)
            return float(stability)
        return 0.0
    
    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "dim": self.dim,
            "token": self.token.tolist(),
            "history": [h.tolist() for h in self.history[-10:]]  # Save recent history
        }
    
    def from_dict(self, data: dict):
        """Restore from checkpoint."""
        self.dim = data["dim"]
        self.token = np.array(data["token"])
        self.history = [np.array(h) for h in data.get("history", [])]
