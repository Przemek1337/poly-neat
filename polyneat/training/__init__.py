"""Shared training machinery: preprocessing, class weights and RNG streams.

This package exists because more than one algorithm in the library now trains
its phenotypes with backpropagation and needs the same non-algorithmic parts:
turning raw images into batches, weighting an imbalanced loss, deriving
independent random streams, and running one training recipe against a model it
knows nothing about.

It deliberately knows nothing about genomes, algorithm names or dataset
classes. Anything that depends on how a particular algorithm learns - EXACT
inheriting its kernels, DeepNEAT reinitializing from scratch - stays in that
algorithm's adapter.
"""
