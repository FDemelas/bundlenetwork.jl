# Hyper-Parameter Learning

This methodology maintains the structure of the Bundle Method, particularly the resolution of the (Quadratic) Master Problem (MP).  
It uses an ML model based on Recursion to compute, at each step, a regularization parameter used as a weight for the Euclidean distance with respect to the best point found so far in the MP.

We assume that you have already downloaded or created your data and saved it in `./data/<your_folder>/`.  
This folder should contain 2000 instances, with 1000 for training, 500 for validation, and 500 for testing.

## Training

Here we show how to train a model.

### Without Batch

```julia
julia --project=. ./runs/runTrainingT.jl --data ./data/<your_folder>/ --lr 1.0e-4 --cn 5 --mti 4 --mvi 4 --seed 1 --maxIT 10 --maxEP 10 --cr_init false --telescopic true --instance_features true --gamma 0.9 --single_prediction false
```

### With Batch

No mechanism to handle batch training is implemented for this model.

## Parameters Explanation

- `--data ./data/<your_folder>/` : The data folder containing the dataset of instances.
- `--lr 1.0e-4` : The learning rate for updating the model parameters.
- `--cn 5` : Coefficient to prevent the gradient norm from becoming too large.
- `--mti 1000` : Number of training instances.
- `--mvi 500` : Number of validation instances.
- `--seed 1` : Random seed.
- `--maxIT 10` : Number of iterations (unrolls) of the Bundle method at each call.
- `--maxEP 10` : Number of training epochs.
- `--cr_init false` : If false, start from the zero vector; otherwise, from the dual solution of the continuous relaxation (CR). Works only in the Lagrangian relaxation (LR) setting, assuming CR is easy to solve but provides a poor bound compared to LR.
- `--telescopic true` : If true, considers a telescopic sum of all visited points during execution.
- `--instance_features true` : If true, adds features depending on the instance (static during execution).
- `--gamma 0.9` : Coefficient in the telescopic sum, larger for the last point, smaller for earlier points. Null Steps have lower contribution.
- `--single_prediction false` : If true, uses only one prediction; otherwise, multiple ones obtained with a recurrent model.

## Testing

```julia
julia --project=. ./runs/runTestT.jl --data ./data/<your_folder>/ --model ./res/res_goldLossWeights_withInstFeat_initZero_lr0.0001_cn5_maxIT10_maxEP10_data<your_folder>_exactGradtrue_gamma0.9_seed1_single_predictionfalse_0.0_ --dataset ./res/BatchVersion_bs_1_true_<your_folder>_0.0001_0.9_5_4_4_1_10_10_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0/
```

### Parameters Explanation
- `--data ./data/<your_folder>/` : The folder containing the dataset.
- `--dataset ./res/BatchVersion_bs_1_true_<your_folder>_0.0001_0.9_5_1000_500_1_10_100_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0/` : The model folder, typically obtained from training.
- `--model` : (Optional) Path to a specific model. If not provided, defaults to the test instances associated with the training. To test a specific set, pass the path to a folder containing a `dataset.json` file, saved during training.

## Further References

- [Bundle Network](#): A fully ML-based Bundle Method.
- [Baselines](#): Grid-search based baselines, including classic (aggregated) Bundle method with different T-strategies (tuning strategies for the regularization parameter), Adam, and Descent.
