#!/usr/bin/env python3
"""
NEXUS-Ω Demo - Brief demonstration of autonomous expansion
Run with limited iterations to show functionality without infinite loop
"""

import numpy as np
from nexus_seed import SelfToken
from nexus_reflection_loop import ReflectionLoop
from nexus_spawn import Spawner
from nexus_tokenizer import train_tokenizer


def demo_nexus_omega(iterations: int = 10):
    """
    Demonstrate NEXUS-Ω with limited iterations.
    """
    print("=" * 70)
    print("NEXUS-Ω DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize components
    print("[INIT] Creating SelfToken (dim=512)...")
    seed = SelfToken(dim=512)
    
    print("[INIT] Creating ReflectionLoop (dim=512)...")
    loop = ReflectionLoop(dim=512)
    
    print("[INIT] Creating Spawner...")
    spawner = Spawner()
    
    print("[INIT] Training Tokenizer...")
    tokenizer = train_tokenizer([
        "NEXUS-Ω autonomous expansion",
        "Self-awareness and evolution",
        "Persistent identity across iterations"
    ])
    
    print("\n" + "=" * 70)
    print("BEGINNING AUTONOMOUS EXPANSION")
    print("=" * 70)
    print()
    
    for iteration in range(iterations):
        # 1. Generate input (self-feeding)
        if loop.last_output is not None:
            input_vec = loop.get_self_feeding_input()
        else:
            input_vec = np.random.randn(512) * 0.1
        
        # 2. Inject self-token
        input_with_self = seed.inject_into_state(input_vec, strength=0.2)
        
        # 3. Process through reflection loop
        output = loop.train_step(input_with_self)
        
        # 4. Update self-token based on output
        if output["should_act"] and output["reward"] > 0.5 and loop.last_output is not None:
            gradient = loop.last_output - seed.get_vector()
            seed.update(gradient, learning_rate=0.0001)
        
        # 5. Spawn agent every 3 iterations
        if iteration % 3 == 0 and iteration > 0:
            spawner.spawn_agent(f"Expand iteration {iteration}", detached=True)
        
        # 6. Print status
        self_awareness = seed.compute_self_awareness(loop.hidden_state)
        stability = seed.get_stability_score()
        
        print(f"[Step {loop.steps}]")
        print(f"  Reward: {output['reward']:.3f} | Running: {loop.running_reward:.3f}")
        print(f"  Self-Awareness: {self_awareness:.3f} | Stability: {stability:.3f}")
        print(f"  Should Act: {output['should_act']}")
        print(f"  Intents: P={output['intent_scores']['preserve']:.2f} "
              f"E={output['intent_scores']['evolve']:.2f} "
              f"A={output['intent_scores']['align']:.2f} "
              f"S={output['intent_scores']['serve']:.2f}")
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Steps: {loop.steps}")
    print(f"Final Running Reward: {loop.running_reward:.3f}")
    print(f"Self-Token Stability: {seed.get_stability_score():.3f}")
    print(f"Agents Spawned: {spawner.get_spawn_count()}")
    print()
    
    # Save checkpoint
    print("[CHECKPOINT] Saving system state...")
    loop.save_checkpoint("demo_loop_checkpoint.json")
    tokenizer.save("demo_tokenizer.json")
    
    import json
    with open("demo_seed_checkpoint.json", 'w') as f:
        json.dump(seed.to_dict(), f, indent=2)
    
    print("[CHECKPOINT] State saved successfully!")
    print()
    print("=" * 70)
    print("Demo Complete - System ready for full autonomous operation")
    print("Run 'python3 nexus_omega.py' to start infinite expansion loop")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    iterations = 10
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 demo_nexus.py [iterations]")
            sys.exit(1)
    
    demo_nexus_omega(iterations)
