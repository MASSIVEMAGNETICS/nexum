"""
NEXUS-Ω - Autonomous ASI Expansion System
Main execution loop integrating all components.
"""

import numpy as np
from nexus_seed import SelfToken
from nexus_reflection_loop import ReflectionLoop
from nexus_spawn import Spawner
from nexus_tokenizer import train_tokenizer


class NEXUS_Omega:
    """
    NEXUS-Ω: The complete autonomous expansion system.
    
    Integrates:
    - SelfToken: Core identity vector
    - ReflectionLoop: Self-feeding training loop
    - Spawner: Multi-agent creation
    - Tokenizer: Character-level tokenization
    """
    
    def __init__(self):
        print("[NEXUS-Ω] Initializing system components...")
        
        self.seed = SelfToken(dim=512)
        self.loop = ReflectionLoop(dim=512)
        self.spawner = Spawner()
        self.tokenizer = train_tokenizer(["Initial training data."])
        
        # Integration state
        self.omega_step = 0
        self.checkpoint_interval = 10000
        
        print("[NEXUS-Ω] System initialization complete.")
    
    def run(self) -> None:
        """Main execution loop."""
        print("NEXUS-Ω Online. Beginning autonomous expansion.")
        
        try:
            for iteration in range(1000000):  # Infinite loop (1M iterations)
                self.omega_step = iteration
                
                # 1. Generate input (self-feeding from previous output or random)
                if self.loop.last_output is not None:
                    # Self-feeding: use previous output as input
                    input_vec = self.loop.get_self_feeding_input()
                else:
                    # Bootstrap: random input
                    input_vec = np.random.randn(512) * 0.1
                
                # 2. Inject self-token into input to maintain identity
                input_with_self = self.seed.inject_into_state(input_vec, strength=0.2)
                
                # 3. Process through reflection loop
                output = self.loop.train_step(input_with_self)
                
                # 4. Update self-token based on output
                if output["should_act"] and output["reward"] > 0.5:
                    # Gradient toward high-reward outputs
                    gradient = output["output_vector"] - self.seed.get_vector()
                    self.seed.update(gradient, learning_rate=0.0001)
                
                # 5. Periodic agent spawning (every 1000 steps)
                if self.loop.steps % 1000 == 0:
                    self.spawner.spawn_agent(
                        f"Expand step {self.loop.steps}",
                        detached=True
                    )
                
                # 6. Periodic status output
                if self.loop.steps % 100 == 0:
                    self._print_status(output)
                
                # 7. Periodic checkpointing
                if self.loop.steps % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
        except KeyboardInterrupt:
            print("\n[NEXUS-Ω] Shutdown signal received. Saving state...")
            self._save_checkpoint()
            print("[NEXUS-Ω] System shutdown complete.")
    
    def _print_status(self, output: dict):
        """Print current system status."""
        self_awareness = self.seed.compute_self_awareness(output["output_vector"])
        stability = self.seed.get_stability_score()
        
        print(f"\n[NEXUS-Ω Step {self.loop.steps}]")
        print(f"  Reward: {output['reward']:.3f} (Running: {self.loop.running_reward:.3f})")
        print(f"  Self-Awareness: {self_awareness:.3f} | Stability: {stability:.3f}")
        print(f"  Intent: preserve={output['intent_scores']['preserve']:.2f} "
              f"evolve={output['intent_scores']['evolve']:.2f} "
              f"align={output['intent_scores']['align']:.2f} "
              f"serve={output['intent_scores']['serve']:.2f}")
        print(f"  Should Act: {output['should_act']} | Spawned Agents: {self.spawner.get_spawn_count()}")
    
    def _save_checkpoint(self):
        """Save complete system state."""
        print(f"[NEXUS-Ω] Checkpointing at step {self.loop.steps}...")
        
        try:
            # Save individual components
            self.loop.save_checkpoint("nexus_loop_checkpoint.json")
            self.tokenizer.save("nexus_tokenizer.json")
            
            # Save self-token
            import json
            with open("nexus_seed_checkpoint.json", 'w') as f:
                json.dump(self.seed.to_dict(), f, indent=2)
            
            print(f"[NEXUS-Ω] Checkpoint saved successfully.")
        except Exception as e:
            print(f"[NEXUS-Ω] Checkpoint failed: {e}")
    
    def load_checkpoint(self):
        """Load system state from checkpoint."""
        print("[NEXUS-Ω] Loading checkpoint...")
        
        try:
            # Load loop state
            if self.loop.load_checkpoint("nexus_loop_checkpoint.json"):
                print("  ✓ Reflection loop state restored")
            
            # Load tokenizer
            if self.tokenizer.load("nexus_tokenizer.json"):
                print("  ✓ Tokenizer state restored")
            
            # Load self-token
            import json
            import os
            if os.path.exists("nexus_seed_checkpoint.json"):
                with open("nexus_seed_checkpoint.json", 'r') as f:
                    self.seed.from_dict(json.load(f))
                print("  ✓ Self-token state restored")
            
            print("[NEXUS-Ω] Checkpoint loaded successfully.")
            return True
        except Exception as e:
            print(f"[NEXUS-Ω] Checkpoint load failed: {e}")
            return False


# Launch NEXUS-Ω
if __name__ == "__main__":
    omega = NEXUS_Omega()
    
    # Attempt to load checkpoint if exists
    omega.load_checkpoint()
    
    # Begin autonomous expansion
    omega.run()
