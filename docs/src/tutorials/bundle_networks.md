# Bundle Network

This methodology substitutes the resolution of the (Quadratic) Master Problem in the Bundle Method, computing the search direction as a convex combination of the sub-gradients at visited points, with an ML model based on Recursion and Attention.

We assume that you have already downloaded or created your data and saved it in `./data/<your_folder>/`.  
This folder should contain 2000 instances, with 1000 used for training, 500 for validation, and 500 for testing.

---

## Training

Here we demonstrate how to train such a model, with or without batching.

### Without Batching

```bash
julia --project=. ./runs/runTraining.jl --data ./data/<your_folder>/ --lr 1.0e-4 --decay 0.9 --cn 5 --mti 1000 --mvi 500 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 2 --always_batch true --h_act softplus --sampling_gamma false
```

### Using Batching

```bash
julia --project=. ./runs/runTraining.jl --data ./data/<your_folder>/ --lr 1.0e-4 --decay 0.9 --cn 5 --mti 100 --mvi 500 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 1 --always_batch true --h_act softplus --sampling_gamma false
```

---

## Parameter Explanation

- `--data ./data/<your_folder>/`: Path to the folder containing the dataset instances.  
- `--lr 1.0e-4`: Learning rate for updating the model parameters.  
- `--decay 0.9`: Decay factor for the optimizer.  
- `--cn 5`: Coefficient to prevent the gradient norm from becoming too large.  
- `--mti 1000`: Number of training instances.  
- `--mvi 500`: Number of validation instances.  
- `--seed 1`: Random seed.  
- `--maxIt 10`: Number of iterations (unrolls) of the Bundle method per call.  
- `--maxEP 10`: Number of training epochs.  
- `--soft_updates true`: If false, uses the standard bundle strategy; if true, uses a smoothed update based on the softplus function.  
- `--h_representation 32`: Size of the hidden representation for each visited point.  
- `--use_softmax false`: If true, uses softmax to predict the convex combination coefficients; otherwise, uses sparsemax.  
- `--gamma 0.999`: Coefficient in the telescopic sum, larger for the last point, smaller for earlier points.  
- `--lambda 0.0`: Final contribution weight: `lambda * final_point + (1 - lambda) * stabilization_point`.  
- `--delta 0.0`: Additional regularization parameter.  
- `--use_graph false`: If true, uses a bipartite graph representation instead of recurrence (still in development; not recommended).  
- `--maxItBack -1`: (Description missing; please clarify if needed.)  
- `--maxItVal 20`: Number of iterations of the Bundle method for validation instances.  
- `--batch_size 1`: Batch size.  
- `--sampling_gamma false`: If true, uses sampling in the hidden representation; otherwise, no.  
- `--always_batch true`: If `batch_size` is 1 and this is true, batching is enforced anyway.  
- `--h_act softplus`: Activation function used in hidden layers.

---

## Testing

To evaluate a pretrained model's performance:

```bash
julia --project=. ./runs/runTest.jl --folder ./data/<your_folder>/ --model_folder BatchVersion_bs_1_true_<your_folder>_0.0001_0.9_5_1000_500_1_10_100_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0
```

---

## Parameters Explanation

- `--data ./data/<your_folder>/`: Path to the dataset folder.  
- `--dataset ./res/BatchVersion_bs_1_true_<your_folder>_0.0001_0.9_5_1000_500_1_10_100_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0/`: Path to the model folder, usually obtained from training.  
- `--model`: (Optional) Path to a specific model folder; if omitted, defaults to the associated test instances folder.

---

## Further References

- [Hyper-Parameter Learning](#): Learning hyperparameters in the (aggregated) Bundle method.  
- [Baselines](#): Grid-search based baselines, including classic (aggregated) Bundle methods with various strategies, Adam, and Gradient Descent.
