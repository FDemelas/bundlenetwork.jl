# Baselines

To compare our approaches with other existing methods, we propose several baselines:
- The classic bundle method with heuristic strategies to tune the regularization parameter. Further details can be found: [https://lipn.univ-paris13.fr/~demelas/Manuscript_Final.pdf](https://lipn.univ-paris13.fr/~demelas/Manuscript_Final.pdf)
- A Flux implementation of the classic gradient descent.
- A Flux implementation of Adam optimizer.

For each baseline, we consider an initial (step-size/regularization) parameter obtained through a grid search, and we save all the results in a JSON file.

Assuming that you have already downloaded or created your data and saved it in `./data/<your_folder>/`, you can run the baselines as:

```julia
julia --project=. ./runs/runBaselines.jl --folder ./data/<your_folder>/ --maxIterDescentType 1000 --maxIterBundle 100 --TS 0.01 0.1 1.0 1 10 100 1000
```

## Parameters Explanation
- `--folder ./data/<your_folder>/`: The folder containing your data.
- `--maxIterDescentType 1000`: The maximum number of iterations for Adam and Descent.
- `--maxIterBundle 100`: The maximum number of iterations for the Bundle methods.
- `--TS 0.01 0.1 1.0 1 10 100 1000`: The initial parameters to consider in the grid search.

### Further References

- [Bundle Network](#): A fully ML-based Bundle Method.
- [Hyper-Parameter Learning](#): Learning an hyper-parameter in the (aggregated) Bundle method.
