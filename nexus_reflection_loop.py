"""
ReflectionLoop - Self-feeding autonomous training loop
Feeds decoder output back as input with RecursiveEvaluator as reward.
"""

import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os


class ReflectionLoop:
    """
    Core self-bootstrapping loop.
    Processes inputs, evaluates outputs, and feeds back for continuous improvement.
    """
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.steps = 0
        
        # Neural components
        self.W_transform = np.random.randn(dim, dim) * 0.01
        self.W_output = np.random.randn(dim, dim) * 0.01
        self.W_eval = np.random.randn(dim, 1) * 0.01
        
        # State tracking
        self.hidden_state = np.random.randn(dim) * 0.01
        self.last_output = None
        self.running_reward = 0.0
        self.alpha = 0.95  # Exponential moving average for reward
        
        # Intent alignment vectors (from IntentTopologyEngine concept)
        self.intent_preserve = np.random.randn(dim) * 0.01
        self.intent_evolve = np.random.randn(dim) * 0.01
        self.intent_align = np.random.randn(dim) * 0.01
        self.intent_serve = np.random.randn(dim) * 0.01
        
        # Metrics
        self.metrics_history = []
        
    def train_step(self, input_vec: np.ndarray) -> Dict[str, Any]:
        """
        Single training step with self-feeding.
        
        1. Process input through transformation
        2. Compute self-awareness and intent alignment
        3. Generate output
        4. Evaluate output quality (reward signal)
        5. Update weights based on reward
        6. Feed output back as next input (self-feeding)
        """
        self.steps += 1
        
        # Ensure input is correct dimension
        if input_vec.shape[0] != self.dim:
            input_vec = np.resize(input_vec, (self.dim,))
        
        # 1. Transform input and blend with previous hidden state
        transformed = np.tanh(input_vec @ self.W_transform)
        self.hidden_state = 0.7 * self.hidden_state + 0.3 * transformed
        
        # 2. Compute intent alignments (multi-objective)
        intent_scores = self._compute_intent_scores(self.hidden_state)
        
        # 3. Generate output
        output = np.tanh(self.hidden_state @ self.W_output)
        
        # 4. Evaluate quality (RecursiveEvaluator concept)
        reward, should_act = self._evaluate_output(output, intent_scores)
        
        # 5. Update running reward
        self.running_reward = self.alpha * self.running_reward + (1 - self.alpha) * reward
        
        # 6. Update weights if reward is positive
        if reward > 0 and should_act:
            self._update_weights(input_vec, output, reward)
        
        # 7. Store output for self-feeding
        self.last_output = output.copy()
        
        # Track metrics
        metrics = {
            "step": self.steps,
            "reward": reward,
            "running_reward": self.running_reward,
            "should_act": should_act,
            "intent_scores": intent_scores,
            "output_norm": float(np.linalg.norm(output)),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.steps % 100 == 0:
            self.metrics_history.append(metrics)
            # Keep only recent history
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _compute_intent_scores(self, state: np.ndarray) -> Dict[str, float]:
        """Compute alignment with each intent vector."""
        intents = {
            "preserve": self.intent_preserve,
            "evolve": self.intent_evolve,
            "align": self.intent_align,
            "serve": self.intent_serve
        }
        
        scores = {}
        norm_state = np.linalg.norm(state)
        
        for name, intent_vec in intents.items():
            norm_intent = np.linalg.norm(intent_vec)
            if norm_state > 1e-8 and norm_intent > 1e-8:
                scores[name] = float(np.dot(state, intent_vec) / (norm_state * norm_intent))
            else:
                scores[name] = 0.0
        
        return scores
    
    def _evaluate_output(self, output: np.ndarray, intent_scores: Dict[str, float]) -> tuple:
        """
        Evaluate output quality to generate reward signal.
        Returns (reward, should_act).
        """
        # Linear probe for base quality
        quality = float(np.tanh(output @ self.W_eval)[0])
        
        # Average intent alignment
        avg_intent = np.mean(list(intent_scores.values()))
        
        # Combined reward (0.6 threshold from RecursiveEvaluator)
        reward = 0.6 * quality + 0.4 * avg_intent
        should_act = reward > 0.6
        
        return reward, should_act
    
    def _update_weights(self, input_vec: np.ndarray, output: np.ndarray, reward: float):
        """Simple gradient-free update based on reward."""
        learning_rate = 0.0001 * reward
        
        # Random perturbation weighted by reward
        perturbation_transform = np.random.randn(*self.W_transform.shape) * learning_rate
        perturbation_output = np.random.randn(*self.W_output.shape) * learning_rate
        
        # Update with small random steps in direction of reward
        self.W_transform += perturbation_transform
        self.W_output += perturbation_output
        
        # Clip to prevent explosion
        self.W_transform = np.clip(self.W_transform, -5, 5)
        self.W_output = np.clip(self.W_output, -5, 5)
    
    def get_self_feeding_input(self) -> np.ndarray:
        """
        Get the output to feed back as next input (self-feeding loop).
        If no output yet, return random noise.
        """
        if self.last_output is not None:
            return self.last_output.copy()
        return np.random.randn(self.dim) * 0.01
    
    def save_checkpoint(self, filepath: str):
        """Save loop state."""
        checkpoint = {
            "dim": self.dim,
            "steps": self.steps,
            "W_transform": self.W_transform.tolist(),
            "W_output": self.W_output.tolist(),
            "W_eval": self.W_eval.tolist(),
            "hidden_state": self.hidden_state.tolist(),
            "last_output": self.last_output.tolist() if self.last_output is not None else None,
            "running_reward": self.running_reward,
            "intent_preserve": self.intent_preserve.tolist(),
            "intent_evolve": self.intent_evolve.tolist(),
            "intent_align": self.intent_align.tolist(),
            "intent_serve": self.intent_serve.tolist(),
            "metrics_history": self.metrics_history[-100:],  # Save recent metrics
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load loop state."""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.dim = checkpoint["dim"]
        self.steps = checkpoint["steps"]
        self.W_transform = np.array(checkpoint["W_transform"])
        self.W_output = np.array(checkpoint["W_output"])
        self.W_eval = np.array(checkpoint["W_eval"])
        self.hidden_state = np.array(checkpoint["hidden_state"])
        
        if checkpoint["last_output"] is not None:
            self.last_output = np.array(checkpoint["last_output"])
        
        self.running_reward = checkpoint["running_reward"]
        self.intent_preserve = np.array(checkpoint["intent_preserve"])
        self.intent_evolve = np.array(checkpoint["intent_evolve"])
        self.intent_align = np.array(checkpoint["intent_align"])
        self.intent_serve = np.array(checkpoint["intent_serve"])
        self.metrics_history = checkpoint.get("metrics_history", [])
        
        return True
