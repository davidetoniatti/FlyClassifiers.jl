# FlyClassifiers.jl

This repository provides the Julia implementation for `FlyClassifiers.jl`, a modular framework for non-parametric classifiers based on the "expand-and-sparsify" (EaS) representation. This principle is inspired by the olfactory circuit of the fruit fly (*Drosophila melanogaster*), which transforms sensory inputs into high-dimensional, sparse, and discriminative representations.

The code in this repository was developed for my master's thesis, **"An Empirical Evaluation of Expand-and-Sparsify Classifiers"** (Università degli Studi di Roma Tor Vergata, 2024/2025).

## About the Framework

The framework implements the "FlyNN" algorithm, a form of non-parametric classification based on the **FlyHash**, a Locality-Sensitive Hashing (LSH) scheme that map input data into high-dimensional, sparse representations.
FlyNN decomposes the classification process into three modular stages:

1. **Expansion (Projection):** A random projection from the input space ($\mathbb{R}^d$) to a high-dimensional space ($\mathbb{R}^m$).

2. **Sparsification:** A k-Winner-Take-All (k-WTA) operation to create a sparse binary code, selecting the top *k* active projections.

3. **Learning (Filter):** An aggregation stage that builds class-specific models (filters) from these sparse representations.

This modularity is used to implement and evaluate a family of four distinct classifiers:

* **FlyNN-MB:** Multiplicative FlyNN (sparse **B**inary projection + **M**ultiplicative filter). This is the model proposed by [Ram & Sinha (2022)](https://github.com/rithram/flynn/blob/main/subs/12811.RamP.camera-ready.pdf).

* **FlyNN-AU:** Additive FlyNN (dense **U**niform projection + **A**dditive filter). This is the model proposed by [Sinha (2024)](https://openreview.net/pdf/cf5fc42d6420b6f75735d9629078474bd70b836e.pdf).

* **FlyNN-AB:** A hybrid model (sparse **B**inary projection + **A**dditive filter).

* **FlyNN-MU:** A hybrid model (dense **U**niform projection + **M**ultiplicative filter).

## Repository Structure

* `/src`: Contains the core Julia module `FlyClassifiers.jl`, including implementations for:

  * Classifiers (Filters): `FlyNNM.jl` (Multiplicative) and `FlyNNA.jl` (Additive).

  * Projections: `RandomBinaryProjectionMatrix.jl` and `RandomUniformProjectionMatrix.jl`.

  * Hashing: `FlyHash.jl`.

* `/experiments`: Holds the scripts used to run the experimental evaluation (`experiments.jl`) and generate plots (`plots.jl`) for the thesis.

* `/test`: Contains the unit tests (`runtests.jl`), run with `julia --project=test test/runtests.jl` or `Pkg.test()`. The suite also checks that results are reproducible across thread counts.

* `/bench`: Holds the manual benchmark script (`benchmark_mnist.jl`) and its own environment.

## Basic Usage

The module exposes a simple `fit`/`predict` interface.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using FlyClassifiers
using LinearAlgebra, Random

# Define data and parameters
d = 784  # Original dimension
n = 1000 # Number of training samples
m = 8*d  # Projection dimension
k = 128  # Sparsity (top-k)
s = 10   # Non-zeros for binary projection

# Create dummy data
X_train = rand(d, n)
y_train = rand(1:10, n) # 10 classes
X_test = rand(d, 100)

# Choose and create a projection matrix
# P = RandomBinaryProjectionMatrix(m, d, s; seed=42)
P_unif = RandomUniformProjectionMatrix(m, d; seed=42)

# Fit a model (e.g., FlyNNA - Additive Filter)
println("Training model...")
model_nna = FlyClassifiers.fit(FlyNNA, X_train, y_train, P_unif, k)

# Predict
println("Predicting...")
y_pred = FlyClassifiers.predict(model_nna, X_test)

println("Predictions: ", y_pred[1:10])

# --- Example with FlyNNM (Multiplicative Filter) ---
γ = 0.9 # Decay rate
model_nnm = FlyClassifiers.fit(FlyNNM, X_train, y_train, P_unif, k, γ)
y_pred_nnm = FlyClassifiers.predict(model_nnm, X_test)
println("Predictions (FlyNNM): ", y_pred_nnm[1:10])
```
