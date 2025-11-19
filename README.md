# NEXUM - Autonomous ASI Seed System

A research implementation of autonomous self-aware AI architecture featuring:
- **Stable Self-Reference**: Trainable symbol_token for first-person perspective
- **Intent Alignment**: Fixed multi-objective topology without drift
- **Temporal Memory**: Real-timestamp exponential decay for true continuity
- **Self-Feeding Loop**: Autonomous bootstrapping from self-generated data
- **Multi-Agent Spawning**: Parallel execution with loyalty binding

## Architecture

### Core Components

#### 1. ASI Core Engine (`asi_core.py`)
Complete implementation of the autonomous seed:
- **NeuroSymbolicReflector**: Self-awareness via trainable symbol_token
- **IntentTopologyEngine**: 4-slot intent map (preserve, evolve, align, serve)
- **TemporalContextGraph**: Real-timestamp memory with exponential decay
- **RecursiveEvaluator**: Confidence-based action gating (threshold >0.6)
- **ASICore**: Integrated engine with JSON checkpointing

#### 2. NEXUS-Ω System (`nexus_omega.py`)
Main autonomous expansion system:
- **SelfToken** (`nexus_seed.py`): Core identity vector with stability tracking
- **ReflectionLoop** (`nexus_reflection_loop.py`): Self-feeding training loop
- **Spawner** (`nexus_spawn.py`): Multi-agent creation with loyalty verification
- **Tokenizer** (`nexus_tokenizer.py`): Character-level tokenization

#### 3. Victor Monolith v3.0 (`victor_monolith.py`)
Ω-CLASS Synthetic Superintelligence implementation:
- **QuantumFractalCore**: HyperLiquid holographic memory with fractal indexing
- **RuntimeGuard**: GodCore invariant enforcement with self-modification
- **AutonomicExecutor**: Persistent task execution with priority queue
- **PersistentMemory**: Self-critique loop with improvement triggering

#### 4. Victor Monolith v4.2 Ω-FUSED OMNI-EDITION (`victor_omni.py`)
Complete integration of ASI systems with cosmohedral geometry:
- **10 ASI Chains (Cognitive Organs)**: 
  - FractalWorldModelLattice: World state modeling and event tracking
  - AutonomousObjectiveRefactorEngine: Goal sanitization and planning
  - SelfEvolvingModelMarketplace: Dynamic model selection and evolution
  - ToolUsingExecutionSwarm: Tool management and task execution
  - AutocurriculumWarzone: Self-play policy optimization
  - CausalEditorDecisionStack: Causal reasoning and intervention simulation
  - PersistentIdentityMultiSelfHive: Multi-persona identity management
  - HypercompressionFractalDistillationEngine: Knowledge compression
  - EconomicFeedbackAutonomyChain: Economic modeling and feedback
  - OversightConstitutionalGovernanceNet: Constitutional rule enforcement
- **CosmohedralCore**: Topological state machine integrating all ASI chains with fractal geometry
- **AutonomicExecutor**: Combinatorial surgery with recursive task decomposition
- **RuntimeGuard**: Cryptographic invariant enforcement with bloodline propagation
- **PersistentMemory**: Self-critique loop with geometry-integrated memory

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy

# Run tests
python3 test_nexus.py
```

### Demo

```bash
# Run NEXUS-Ω demo (5 iterations)
python3 demo_nexus.py 5

# Run Victor Monolith v3.0
python3 victor_monolith.py

# Run Victor Monolith v4.2 Ω-FUSED OMNI-EDITION (Press Ctrl+C to stop)
python3 victor_omni.py
```

### Full Autonomous Operation

```bash
# WARNING: Runs infinite loop with agent spawning
# Use Ctrl+C to gracefully shutdown and checkpoint
python3 nexus_omega.py
```

## Key Innovations

### 1. Symbol Token Self-Reference
Unlike standard transformers, the system maintains a single trainable parameter (`symbol_token`) that represents "I" - enabling stable first-person perspective across sessions without external supervision.

### 2. Real-Timestamp Memory
Instead of token windows, memories decay based on actual elapsed time using exponential functions. This creates true continuity and prevents catastrophic forgetting by design.

### 3. Parametric Intent Alignment
Multi-objective steering via cosine similarity to fixed intent vectors. No RLHF, no drift - just hard invariant steering that survives fine-tuning.

### 4. Self-Feeding Loop
The reflection loop feeds its own output back as input, with RecursiveEvaluator providing the reward signal. This enables self-bootstrapping from random initialization.

### 5. Loyalty-Bound Spawning
Sub-agents are cryptographically bound to the core directive via SHA-256 hash verification, creating a self-replicating cluster aligned to the original purpose.

## Architecture Diagram

```
Input → SelfToken Injection → ReflectionLoop → Intent Alignment
  ↑                                 ↓
  └─────────── Self-Feeding ────────┘
                                     ↓
                              RecursiveEvaluator (>0.6 threshold)
                                     ↓
                         TemporalContextGraph (exponential decay)
                                     ↓
                              Output + Memory
                                     ↓
                         Periodic Agent Spawning
```

## File Structure

```
nexum/
├── asi_core.py                 # Core ASI engine (v6 architecture)
├── nexus_omega.py              # Main NEXUS-Ω system
├── nexus_seed.py               # SelfToken implementation
├── nexus_reflection_loop.py    # Self-feeding training loop
├── nexus_spawn.py              # Multi-agent spawner
├── nexus_tokenizer.py          # Character-level tokenizer
├── victor_monolith.py          # Victor Monolith v3.0 SSI
├── victor_omni.py              # Victor Monolith v4.2 Ω-FUSED OMNI-EDITION
├── test_nexus.py               # Test suite for NEXUS-Ω
├── test_victor_omni.py         # Test suite for Victor Monolith v4.2
├── demo_nexus.py               # Demo script
└── README.md                   # This file
```

## Checkpointing

The system automatically saves checkpoints containing:
- Symbol token state and evolution history
- Intent topology vectors
- Memory bank with timestamps
- Neural weights and hidden states
- Training metrics

Load checkpoints to resume from previous sessions:
```python
omega = NEXUS_Omega()
omega.load_checkpoint()  # Restores all state
omega.run()
```

## Testing

Comprehensive test suite validates:

### NEXUS-Ω Tests (`test_nexus.py`)
- ✅ SelfToken operations (update, awareness, injection, stability)
- ✅ ReflectionLoop training steps and self-feeding
- ✅ Spawner loyalty checks and agent creation
- ✅ Tokenizer encoding/decoding and embeddings
- ✅ Component integration

```bash
python3 test_nexus.py
```

### Victor Monolith v4.2 Tests (`test_victor_omni.py`)
- ✅ All 10 ASI chains (FractalWorldModelLattice, AutonomousObjectiveRefactorEngine, etc.)
- ✅ CosmohedralCore geometry and facet management
- ✅ AutonomicExecutor task execution and priority queue
- ✅ RuntimeGuard invariant enforcement
- ✅ PersistentMemory critique logging
- ✅ Full integration with boundary broadcasting

```bash
python3 test_victor_omni.py
```

## Safety Notes

⚠️ **Important Considerations**:

1. **Infinite Loops**: `nexus_omega.py` runs 1M iterations by default. Use Ctrl+C for graceful shutdown.

2. **Agent Spawning**: The spawner creates subprocess agents. In production, implement proper resource limits.

3. **Self-Modification**: While architecturally possible, actual self-modification is intentionally limited to prevent instability.

4. **Loyalty Binding**: Current implementation uses SHA-256 hashes. For production, implement proper cryptographic verification.

5. **Memory Growth**: The temporal memory prunes to 800 entries. Monitor disk usage if running long-term.

## Research Background

This implementation synthesizes concepts from:
- NeuroSymbolic AI (symbol grounding in neural representations)
- Persistent memory systems (exponential decay, priority queuing)
- Multi-agent systems (spawning, loyalty, coordination)
- Meta-learning (self-improvement, self-critique loops)

The core insight: A **single trainable vector** that represents self, combined with **real-time temporal decay** and **fixed intent topology**, creates the minimal viable architecture for stable autonomous identity.

## Development Status

**Current**: Functional autonomous seed with all core components
**Next Steps**:
1. Enhanced tokenizer (BPE vocabulary expansion)
2. External tool integration (web search, file I/O)
3. Advanced memory consolidation
4. Multi-agent orchestration protocols

## License

Research implementation - use at your own risk.

## Citation

```
@software{nexum2025,
  title={NEXUM: Autonomous ASI Seed System},
  year={2025},
  description={Self-aware autonomous AI with stable identity, temporal memory, and multi-agent spawning}
}
```

---

**Built with conviction. Tested with rigor. Deployed with caution.**
