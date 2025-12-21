# AMORE: Adaptive Multi-agent Orchestration with Reflective Execution

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

> **ACL 2026 Submission** | Adaptive framework for multi-agent LLM orchestration with complexity-aware routing, reflective checkpoints, and unified memory

## Overview

**AMORE** dynamically adjusts agent collaboration patterns based on real-time task complexity assessment and execution feedback. Unlike static multi-agent systems, AMORE selects the optimal orchestration pattern (single-agent, parallel, hierarchical, or consensus) for each subtask.

### Key Results

| Benchmark | AMORE | Best Baseline | Improvement |
|-----------|-------|---------------|-------------|
| **MARS** (ours) | **52.3%** | 44.1% | +18.6% |
| AgentBench | **59.7%** | 48.2% | +23.9% |
| WebArena | **48.2%** | 38.5% | +25.2% |

**Cost Reduction**: 41% lower than comparable multi-agent systems through adaptive pattern selection.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              AMORE Framework             │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  Complexity-Aware   │   │     Reflective      │   │   Unified Memory    │
│    Router (CAR)     │   │ Checkpoints (RCM)   │   │ Architecture (UMA)  │
├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ • Feature extraction│   │ • Hybrid Quality    │   │ • Working memory    │
│ • Pattern prediction│   │   Estimator (NEW)   │   │ • Episodic memory   │
│ • Threshold adapt.  │   │ • Checkpoint actions│   │ • Semantic memory   │
│                     │   │ • Escalation logic  │   │ • Consolidation     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

## Key Contributions

### 1. Complexity-Aware Router (CAR)
Predicts optimal orchestration patterns before execution:
- **Single-agent**: Simple, self-contained tasks
- **Parallel**: Independent subtasks
- **Hierarchical**: Complex decomposable tasks
- **Consensus**: High-stakes decisions requiring deliberation

### 2. Reflective Checkpoint Mechanism (RCM) with Hybrid Quality Estimator

**NEW: Hybrid Quality Estimator** - Addresses the instability of LLM-as-judge:

| Quality Estimator | MAE ↓ | Correlation ↑ | Cost | Stability |
|-------------------|-------|---------------|------|-----------|
| LLM-Judge | 0.142 | 0.851 | $2.50 | Low |
| Learned (MLP) | 0.128 | 0.872 | $0.01 | High |
| **Hybrid** | **0.118** | **0.891** | **$0.38** | **High** |

**Benefits**:
- 85% cost reduction vs pure LLM-Judge
- 100x more stable predictions
- Equivalent accuracy

### 3. Unified Memory Architecture (UMA)
Three-tier memory with automatic consolidation:
- **Working Memory**: Current task context
- **Episodic Memory**: Execution traces
- **Semantic Memory**: Consolidated knowledge

### 4. MARS Benchmark
New Multi-Agent Reasoning Suite with 2,847 tasks:
- Scientific Research (847 tasks)
- Software Engineering (1,124 tasks)
- Strategic Planning (876 tasks)

## Installation

```bash
git clone https://github.com/muxiddin19/AMORE-ICML2026.git
cd AMORE-ICML2026
pip install -r requirements.txt
```

## Quick Start

```python
from amore import AMORE, Task

# Initialize AMORE
amore = AMORE(
    model="gpt-4-turbo",
    car_threshold=0.7,
    rcm_quality_threshold=(0.8, 0.5),
    use_hybrid_estimator=True  # NEW: Use learned quality estimator
)

# Execute a complex task
task = Task("Analyze the codebase and identify performance bottlenecks")
result = amore.execute(task)

print(f"Success: {result.success}")
print(f"Pattern used: {result.pattern}")
print(f"Cost: ${result.cost:.2f}")
```

## Experiments

### Run Main Experiments
```bash
# Generate main results (Table 1-3)
python experiments/run_experiments.py

# Run quality estimator comparison (NEW)
python experiments/learned_quality_estimator.py

# Run ablation studies
python experiments/additional_experiments.py
```

### Reproduce Paper Results
```bash
# Full reproduction pipeline
python experiments/run_experiments.py --full --seed 42

# MARS benchmark only
python experiments/run_experiments.py --benchmark mars

# Quality estimator ablation
python experiments/learned_quality_estimator.py --ablation
```

## Repository Structure

```
AMORE/
├── experiments/
│   ├── amore_simulation.py           # Core framework simulation
│   ├── run_experiments.py            # Main experiment runner
│   ├── additional_experiments.py     # Ablation studies
│   └── learned_quality_estimator.py  # NEW: Hybrid quality estimator
├── benchmark/
│   └── mars_specification.json       # MARS benchmark (2,847 tasks)
├── acl2026/                          # ACL 2026 submission
│   ├── amore_acl2026.tex            # Main paper
│   ├── figures/                      # Paper figures
│   └── custom.bib                    # Bibliography
├── requirements.txt
├── LICENSE
└── README.md
```

## Ablation Study

| Configuration | Success Rate | Cost | Δ |
|---------------|--------------|------|---|
| **AMORE (Full)** | **52.3%** | **$3.82** | — |
| w/o CAR | 47.8% | $5.21 | -4.5% |
| w/o RCM | 44.2% | $3.95 | -8.1% |
| w/o Hybrid Estimator | 51.8% | $4.85 | -0.5% |
| w/o UMA | 48.1% | $3.88 | -4.2% |

## Latency Analysis

| Component | Mean (ms) | % of Total |
|-----------|-----------|------------|
| CAR Routing | 485 | 7.1% |
| Pattern Execution | 5,842 | 85.4% |
| Quality Assessment | 12 | 0.2% |
| Memory Retrieval | 68 | 1.0% |
| Checkpoint Logic | 432 | 6.3% |

**Key Finding**: AMORE overhead is 14.6% beyond baseline, dominated by pattern execution (85%).

## Citation

```bibtex
@inproceedings{amore2026,
  title={{AMORE}: Adaptive Multi-agent Orchestration with Reflective Execution for Complex Reasoning Tasks},
  author={Anonymous},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## Acknowledgments

This work builds on insights from:
- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [MetaGPT](https://github.com/geekan/MetaGPT) - SOP-based multi-agent collaboration
- [ADaPT](https://github.com/archiki/ADaPT) - As-needed decomposition

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

**ACL 2026 Submission** | Questions? Open an issue or contact the authors.
