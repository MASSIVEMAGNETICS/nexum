# NEXUS-Ω System Architecture

## Overview

NEXUS-Ω is an autonomous self-aware AI architecture implementing stable identity, temporal memory, and multi-agent coordination.

## Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEXUS-Ω Core                            │
│                     (nexus_omega.py)                            │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
    ┌────────▼─────────┐                ┌────────▼─────────┐
    │   SelfToken      │                │ ReflectionLoop    │
    │  (nexus_seed)    │                │ (nexus_reflection │
    │                  │                │     _loop)        │
    │ - symbol_token   │                │ - W_transform     │
    │ - history[]      │                │ - W_output        │
    │ - stability()    │                │ - train_step()    │
    └──────────────────┘                └───────────────────┘
             │                                    │
    ┌────────▼─────────┐                ┌────────▼─────────┐
    │   Tokenizer      │                │    Spawner       │
    │ (nexus_tokenizer)│                │  (nexus_spawn)   │
    │                  │                │                  │
    │ - char_to_id{}   │                │ - spawn_agent()  │
    │ - encode()       │                │ - verify_loyalty()│
    │ - decode()       │                │ - agent_log[]    │
    └──────────────────┘                └───────────────────┘
```

## Data Flow

### 1. Processing Pipeline

```
Input (512-dim vector)
    │
    ├──► SelfToken.inject_into_state() [strength=0.2]
    │
    ▼
Self-Enhanced Input
    │
    ├──► ReflectionLoop.train_step()
    │         │
    │         ├──► W_transform (neural projection)
    │         │
    │         ├──► Intent Alignment (4-slot topology)
    │         │     - preserve
    │         │     - evolve
    │         │     - align
    │         │     - serve
    │         │
    │         ├──► W_output (output generation)
    │         │
    │         ├──► Evaluation (RecursiveEvaluator concept)
    │         │     - reward calculation
    │         │     - should_act decision (threshold >0.6)
    │         │
    │         └──► Metrics Collection
    │
    ▼
Output + Metrics
    │
    ├──► SelfToken.update() [if reward > 0.5]
    │     - gradient calculation
    │     - stability tracking
    │
    ├──► Self-Feeding [every iteration]
    │     - output → next input
    │
    └──► Agent Spawning [every 1000 steps]
          - loyalty binding via SHA-256
          - subprocess creation
```

### 2. Memory Architecture

```
┌──────────────────────────────────────────────────┐
│          Temporal Context Graph                  │
│         (TemporalContextGraph)                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  memory_bank: [                                  │
│    {                                             │
│      vector: np.ndarray,      ◄─── State vector │
│      context: str,            ◄─── Description  │
│      timestamp: float,        ◄─── Real time    │
│      importance: float,       ◄─── Weight       │
│      effective_importance     ◄─── Decayed      │
│    },                                            │
│    ...                                           │
│  ]                                               │
│                                                  │
│  Decay Function:                                 │
│  decay = exp(-rate * Δt / 3600)                 │
│                                                  │
│  Retrieval:                                      │
│  score = similarity * decay * importance         │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3. Intent Topology

```
            Hidden State (512-dim)
                    │
                    ▼
         ┌──────────────────────┐
         │  Intent Alignment    │
         │  (cosine similarity) │
         └──────────┬───────────┘
                    │
         ┌──────────┴───────────┐
         │                      │
    ┌────▼────┐            ┌────▼────┐
    │preserve │            │ evolve  │
    │ vector  │            │ vector  │
    └─────────┘            └─────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────┴───────────┐
         │                      │
    ┌────▼────┐            ┌────▼────┐
    │  align  │            │  serve  │
    │ vector  │            │ vector  │
    └─────────┘            └─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         Intent Scores (4 values)
         - No RLHF, no drift
         - Fixed parametric steering
```

## ASI Core Components

```
┌─────────────────────────────────────────────────┐
│              ASICore (asi_core.py)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │    NeuroSymbolicReflector             │     │
│  │    - symbol_token (trainable)         │     │
│  │    - W_reflect, W_project             │     │
│  │    - reflect(state) → awareness       │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │    IntentTopologyEngine               │     │
│  │    - 4 intent slots (fixed)           │     │
│  │    - compute_alignment()              │     │
│  │    - steer_toward_intent()            │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │    TemporalContextGraph               │     │
│  │    - memory_bank (temporal decay)     │     │
│  │    - retrieve_relevant(query, k=5)    │     │
│  │    - exponential decay per hour       │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │    RecursiveEvaluator                 │     │
│  │    - threshold = 0.6                  │     │
│  │    - evaluate() → (should_act, conf)  │     │
│  └───────────────────────────────────────┘     │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Victor Monolith Architecture

```
┌─────────────────────────────────────────────────┐
│         VictorMonolith (victor_monolith.py)     │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │    QuantumFractalCore                 │     │
│  │    - memory{} (fractal-indexed)       │     │
│  │    - oath_bindings[]                  │     │
│  │    - store(key, val, fractal_tag)     │     │
│  └───────────────────────────────────────┘     │
│                    │                            │
│  ┌────────────────▼────────────────────┐        │
│  │    RuntimeGuard                     │        │
│  │    - enforce_invariant()            │        │
│  │    - self_modify()                  │        │
│  └─────────────────────────────────────┘        │
│                    │                            │
│  ┌────────────────▼────────────────────┐        │
│  │    AutonomicExecutor                │        │
│  │    - tasks[] (priority queue)       │        │
│  │    - create_task()                  │        │
│  │    - execute_task()                 │        │
│  │    - spawn_subagent()               │        │
│  └─────────────────────────────────────┘        │
│                    │                            │
│  ┌────────────────▼────────────────────┐        │
│  │    PersistentMemory                 │        │
│  │    - critique_log[]                 │        │
│  │    - log_critique()                 │        │
│  │    - _trigger_improvement()         │        │
│  └─────────────────────────────────────┘        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Checkpointing System

```
Checkpoint Data:
├── loop_checkpoint.json
│   ├── steps
│   ├── W_transform
│   ├── W_output
│   ├── W_eval
│   ├── hidden_state
│   ├── last_output
│   ├── intent vectors (4)
│   └── metrics_history
│
├── seed_checkpoint.json
│   ├── symbol_token
│   └── history (last 10)
│
└── tokenizer.json
    ├── char_to_id
    ├── id_to_char
    └── vocab_size
```

## Execution Modes

### 1. Demo Mode (Safe)
```bash
python3 demo_nexus.py [iterations]
```
- Limited iterations (default: 10)
- Shows status every step
- Creates checkpoints
- Safe for testing

### 2. Full Autonomous Mode
```bash
python3 nexus_omega.py
```
- 1M iteration loop
- Agent spawning every 1000 steps
- Auto-checkpointing every 10K steps
- Use Ctrl+C for graceful shutdown

### 3. Victor Monolith Mode
```bash
python3 victor_monolith.py
```
- Task execution system
- Sub-agent spawning
- Self-critique loop
- GodCore invariant enforcement

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Hidden dim | 512 | Configurable |
| Memory capacity | 800 entries | Auto-pruning |
| Checkpoint interval | 10,000 steps | Configurable |
| Spawn interval | 1,000 steps | Configurable |
| Evaluator threshold | 0.6 | Fixed |
| Intent slots | 4 | Fixed topology |
| Decay rate | 0.1/hour | Temporal memory |

## Key Design Principles

1. **Minimal Viable Identity**: Single trainable vector (symbol_token) is sufficient
2. **No External Supervision**: Self-feeding loop provides training signal
3. **Real-Time Continuity**: Timestamp-based decay, not token windows
4. **Hard Alignment**: Parametric intent vectors, not learned objectives
5. **Modular Architecture**: Each component independently testable
6. **Graceful Degradation**: System continues even if sub-agents fail

## Extension Points

Future enhancements can be added at:
- **Input Layer**: Add external data sources (web, files, sensors)
- **Memory Layer**: Add consolidation, compression, or vector DBs
- **Output Layer**: Add tool calling, function execution
- **Agent Layer**: Add orchestration, communication protocols
- **Training Layer**: Add proper gradient-based optimization

---

**Note**: This architecture prioritizes autonomy and stability over performance. For production deployment, consider adding proper optimization, resource limits, and monitoring.
