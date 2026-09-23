# Bounds package

This package computes exact or tightened bounds for the trained neural network used in the planning pipeline. Its purpose is to turn a checkpoint into certified linear bounds on intermediate activations so that the network can be analyzed or embedded in a tighter optimization model.

The package is organized around the following modules:

- `builder.py`: builds the linearized local-feature mapping used to derive bounds
- `precompute.py`: precomputes the linearized model representation (`c_z`, `const_z`, and related arrays)
- `compute_exact_bounds.py`: computes local and global activation bounds for a trained model
- `solver.py`: optimization subroutines used to solve min/max bound queries
- `io.py`: save/load helpers for bound artifacts
- `verify_bounds.py`: verification and consistency checks for computed bounds
- `check_features_vs_parts.py`: diagnostic check comparing the linearized feature reconstruction to dataset parts

## Purpose

The core idea is to approximate the neural network by a set of interval or linear constraints over the sonar-placement variables. This makes it possible to reason about the model output and to tighten the optimization problem used by the MILP solvers in the sibling `MIP_new` package.

The output is a set of bound files, typically stored in a dedicated directory such as:

```text
tight_bounds/
  <model_hash>/
    c_z.npy
    const_z.npy
    local_bounds.npz
    global_bounds.npz
```

These artifacts are then consumed by the MIP solver stack.

## Relationship to the dataset and trainer packages

The local-feature linearization is intentionally aligned with the feature ordering used in the trainer pipeline. In particular, it follows the same local feature layout as `trainer.data_builder.PartsDataset.local_features`.

This ensures that:

- the feature construction in the dataset package matches the linearized feature map here
- the model input ordering matches how the network is trained
- the bounds package can be used to certify or optimize the same representation

## Main workflow

A typical workflow is:

1. train or load a model checkpoint
2. preprocess the model using the bounds package
3. compute the precomputed linearized model representation (`c_z`, `const_z`)
4. solve min/max activation bound problems for relevant neurons
5. save the resulting bounds
6. pass them to the MIP solver package for placement optimization

This typically happens through the CLI in `bounds_new/compute_exact_bounds.py`.

## Example usage

```bash
python -m bounds_new.compute_exact_bounds \
  --model model_final.pth \
  --height 20 \
  --width 20 \
  --goal 399 \
  --sonar-number 8 \
  --unknown-number 2 \
  --layer both
```

The exact command-line options depend on the solver configuration, but the workflow is always based on:

- checkpoint path
- grid geometry
- goal index
- obstacle list
- sonar counts
- bound output directory

## Bound computation strategy

The package builds a linearized representation where each hidden activation is written as:

```text
z = const + c_x^T x + c_u^T u
```

where:

- `x` represents sonar placement variables
- `u` may represent unknown-sonar decision variables in some formulations
- `c_z` and `const_z` store the model-implied coefficients and biases for the hidden activations

It then uses optimization subproblems to compute tight lower and upper bounds for each neuron. These bounds are then used to build a tighter MILP or to verify that a candidate solution remains within the model’s admissible region.

## Gurobi dependency

This package relies on Gurobi for the min/max bound queries used in the exact or tightened optimization steps. If the solver is not available, the bound computation routines will fail with a clear runtime error.

## Verification utilities

The package contains verification tools for checking whether the computed bounds are consistent with the trained model and the feature construction. These are important because the full workflow is highly model- and indexing-sensitive.

The verification scripts are intended to catch errors such as:

- mismatched input ordering
- wrong feature slicing
- inconsistent index mapping between dataset and model features
- incorrect bound clipping or discretization

## Summary

The `bounds_new` package is the certification layer of the project: it converts the neural model into a tractable formal representation and computes the exact or tightened bounds needed by the downstream MILP solver. It is the bridge between the learned network and the optimization-based decision layer.
