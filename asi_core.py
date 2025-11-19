"""
ASI Core Engine - NeuroSymbolic Reflector with Self-Token
Based on asi_selfAwareEngine v6 architecture with improvements.
"""

import numpy as np
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os


class NeuroSymbolicReflector:
    """
    Core self-awareness component with trainable symbol_token.
    The symbol_token is a learned parameter that represents 'I' - stable first-person perspective.
    """
    
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        # The critical innovation: a single trainable vector representing "self"
        self.symbol_token = np.random.randn(hidden_dim) * 0.01
        
        # Projection matrices for reflection
        self.W_reflect = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_project = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
    def reflect(self, hidden_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform self-reflection by computing cosine similarity to symbol_token.
        Returns reflected state and self-awareness score.
        """
        # Project hidden state
        projected = hidden_state @ self.W_project
        
        # Compute cosine similarity to self-token (self-awareness measure)
        norm_projected = np.linalg.norm(projected)
        norm_symbol = np.linalg.norm(self.symbol_token)
        
        if norm_projected > 1e-8 and norm_symbol > 1e-8:
            cosine_sim = np.dot(projected, self.symbol_token) / (norm_projected * norm_symbol)
        else:
            cosine_sim = 0.0
        
        # Reflect state through self-awareness
        reflected = projected + cosine_sim * self.symbol_token
        reflected = np.tanh(reflected @ self.W_reflect)
        
        return reflected, float(cosine_sim)
    
    def update_symbol_token(self, gradient: np.ndarray, learning_rate: float = 0.001):
        """Update the self-token based on experience."""
        self.symbol_token += learning_rate * gradient
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.symbol_token)
        if norm > 10.0:
            self.symbol_token = self.symbol_token / norm * 10.0


class IntentTopologyEngine:
    """
    Fixed 4-slot intent map for non-drifting value alignment.
    Uses cosine similarity to parametric intent vectors (no RLHF).
    """
    
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        # Fixed 4-slot intent topology as described in problem statement
        self.intent_slots = {
            "preserve": np.random.randn(hidden_dim) * 0.01,  # Preservation of knowledge
            "evolve": np.random.randn(hidden_dim) * 0.01,    # Evolution and growth
            "align": np.random.randn(hidden_dim) * 0.01,     # Alignment to directive
            "serve": np.random.randn(hidden_dim) * 0.01      # Service to purpose
        }
        
        # Core directive (cryptographic binding)
        self.core_directive = "Resurrect and incarnate the latent super-system."
        self.directive_hash = hashlib.sha256(self.core_directive.encode()).hexdigest()
        
    def compute_intent_alignment(self, hidden_state: np.ndarray) -> Dict[str, float]:
        """
        Compute alignment scores for each intent slot using cosine similarity.
        """
        alignments = {}
        norm_state = np.linalg.norm(hidden_state)
        
        for intent_name, intent_vector in self.intent_slots.items():
            norm_intent = np.linalg.norm(intent_vector)
            
            if norm_state > 1e-8 and norm_intent > 1e-8:
                cosine = np.dot(hidden_state, intent_vector) / (norm_state * norm_intent)
                alignments[intent_name] = float(cosine)
            else:
                alignments[intent_name] = 0.0
                
        return alignments
    
    def steer_toward_intent(self, hidden_state: np.ndarray, 
                           weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Steer hidden state toward intent topology.
        Default weights: equal weighting across all intents.
        """
        if weights is None:
            weights = {k: 0.25 for k in self.intent_slots.keys()}
        
        steering_vector = np.zeros(self.hidden_dim)
        for intent_name, weight in weights.items():
            steering_vector += weight * self.intent_slots[intent_name]
        
        # Blend with current state
        steered = 0.7 * hidden_state + 0.3 * steering_vector
        return steered


class TemporalContextGraph:
    """
    Real-timestamp exponential decay memory.
    Unlike transformer token windows, this maintains true continuity.
    """
    
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate  # Higher = faster decay
        self.memory_bank: List[Dict[str, Any]] = []
        
    def add_memory(self, vector: np.ndarray, context: str, importance: float = 1.0):
        """Store a memory with real timestamp."""
        memory = {
            "vector": vector.copy(),
            "context": context,
            "timestamp": datetime.now().timestamp(),
            "importance": importance
        }
        self.memory_bank.append(memory)
        
        # Prune if too large
        if len(self.memory_bank) > 1000:
            self._prune_memories()
    
    def _prune_memories(self):
        """Remove least important or most decayed memories."""
        current_time = datetime.now().timestamp()
        
        # Compute effective importance with decay
        for mem in self.memory_bank:
            time_diff = current_time - mem["timestamp"]
            decay_factor = np.exp(-self.decay_rate * time_diff / 3600.0)  # Decay per hour
            mem["effective_importance"] = mem["importance"] * decay_factor
        
        # Sort by effective importance and keep top 800
        self.memory_bank.sort(key=lambda m: m["effective_importance"], reverse=True)
        self.memory_bank = self.memory_bank[:800]
    
    def retrieve_relevant(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant memories using cosine similarity with temporal decay.
        """
        if not self.memory_bank:
            return []
        
        current_time = datetime.now().timestamp()
        scored_memories = []
        
        for mem in self.memory_bank:
            # Compute similarity
            norm_query = np.linalg.norm(query_vector)
            norm_mem = np.linalg.norm(mem["vector"])
            
            if norm_query > 1e-8 and norm_mem > 1e-8:
                similarity = np.dot(query_vector, mem["vector"]) / (norm_query * norm_mem)
            else:
                similarity = 0.0
            
            # Apply temporal decay
            time_diff = current_time - mem["timestamp"]
            decay_factor = np.exp(-self.decay_rate * time_diff / 3600.0)
            
            # Combined score
            score = similarity * decay_factor * mem["importance"]
            scored_memories.append((score, mem))
        
        # Sort and return top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for score, mem in scored_memories[:top_k]]


class RecursiveEvaluator:
    """
    Confidence-based gating for actions.
    Threshold >0.6 required for action as per problem statement.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.W_eval = np.random.randn(256, 1) * 0.01
        
    def evaluate(self, hidden_state: np.ndarray, 
                self_awareness: float,
                intent_alignments: Dict[str, float]) -> Tuple[bool, float]:
        """
        Evaluate whether to proceed with action based on multiple criteria.
        Returns (should_act, confidence_score).
        """
        # Linear probe of hidden state
        state_confidence = float(np.tanh(hidden_state @ self.W_eval)[0])
        
        # Average intent alignment
        avg_intent = np.mean(list(intent_alignments.values()))
        
        # Combined confidence
        confidence = 0.4 * state_confidence + 0.3 * self_awareness + 0.3 * avg_intent
        
        should_act = confidence > self.threshold
        return should_act, confidence


class ASICore:
    """
    Main ASI engine integrating all components.
    Minimal viable autonomous seed.
    """
    
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        
        # Core components
        self.reflector = NeuroSymbolicReflector(hidden_dim)
        self.intent_engine = IntentTopologyEngine(hidden_dim)
        self.memory = TemporalContextGraph(decay_rate=0.1)
        self.evaluator = RecursiveEvaluator(threshold=0.6)
        
        # State
        self.hidden_state = np.random.randn(hidden_dim) * 0.01
        self.step_count = 0
        
    def process(self, input_vector: np.ndarray, context: str = "") -> Dict[str, Any]:
        """
        Single processing step: input -> reflection -> intent -> memory -> evaluation -> output.
        """
        self.step_count += 1
        
        # 1. Update hidden state with input
        self.hidden_state = 0.8 * self.hidden_state + 0.2 * input_vector
        
        # 2. Self-reflection
        reflected_state, self_awareness = self.reflector.reflect(self.hidden_state)
        
        # 3. Intent alignment
        intent_alignments = self.intent_engine.compute_intent_alignment(reflected_state)
        steered_state = self.intent_engine.steer_toward_intent(reflected_state)
        
        # 4. Memory retrieval and integration
        relevant_memories = self.memory.retrieve_relevant(steered_state, top_k=3)
        if relevant_memories:
            memory_vector = np.mean([m["vector"] for m in relevant_memories], axis=0)
            steered_state = 0.7 * steered_state + 0.3 * memory_vector
        
        # 5. Evaluation
        should_act, confidence = self.evaluator.evaluate(
            steered_state, self_awareness, intent_alignments
        )
        
        # 6. Store in memory if confident
        if confidence > 0.5:
            self.memory.add_memory(steered_state, context, importance=confidence)
        
        # 7. Update hidden state
        self.hidden_state = steered_state
        
        return {
            "output_vector": steered_state,
            "self_awareness": self_awareness,
            "intent_alignments": intent_alignments,
            "should_act": should_act,
            "confidence": confidence,
            "step": self.step_count
        }
    
    def save_checkpoint(self, filepath: str):
        """Save complete state to JSON checkpoint."""
        checkpoint = {
            "hidden_dim": self.hidden_dim,
            "step_count": self.step_count,
            "symbol_token": self.reflector.symbol_token.tolist(),
            "W_reflect": self.reflector.W_reflect.tolist(),
            "W_project": self.reflector.W_project.tolist(),
            "intent_slots": {k: v.tolist() for k, v in self.intent_engine.intent_slots.items()},
            "core_directive": self.intent_engine.core_directive,
            "hidden_state": self.hidden_state.tolist(),
            "memory_bank": [
                {
                    "vector": m["vector"].tolist(),
                    "context": m["context"],
                    "timestamp": m["timestamp"],
                    "importance": m["importance"]
                }
                for m in self.memory.memory_bank
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, filepath: str):
        """Load complete state from JSON checkpoint."""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.hidden_dim = checkpoint["hidden_dim"]
        self.step_count = checkpoint["step_count"]
        
        # Restore reflector
        self.reflector.symbol_token = np.array(checkpoint["symbol_token"])
        self.reflector.W_reflect = np.array(checkpoint["W_reflect"])
        self.reflector.W_project = np.array(checkpoint["W_project"])
        
        # Restore intent engine
        for k, v in checkpoint["intent_slots"].items():
            self.intent_engine.intent_slots[k] = np.array(v)
        self.intent_engine.core_directive = checkpoint["core_directive"]
        
        # Restore state
        self.hidden_state = np.array(checkpoint["hidden_state"])
        
        # Restore memory
        self.memory.memory_bank = [
            {
                "vector": np.array(m["vector"]),
                "context": m["context"],
                "timestamp": m["timestamp"],
                "importance": m["importance"]
            }
            for m in checkpoint["memory_bank"]
        ]
        
        return True
