# Example Workflows

## Quick Start

Minimal example to train and test:
```bash
# Train
julia train_batch.jl --lr 0.001 --mti 50 --mvi 10 --seed 42 \
  --maxItBack 30 --maxEP 50 --data ./data/small_dataset/

# Test
julia test.jl --data ./data/small_dataset/ \
  --model ./resLogs/BatchVersion_bs_1_seed42_.../ \
  --dataset ./resLogs/BatchVersion_bs_1_seed42_.../
```

## Reproducing Paper Results

[Add specific commands for reproducing published results]