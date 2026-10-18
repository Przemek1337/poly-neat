"""Shared execution machinery for the benchmark example families.

Three example families run the same neuroevolution algorithms under the same
rigor: the pediatric pneumonia X-ray benchmark, the MNIST DeepNEAT/EXACT
comparison and the CIFAR-10 DeepNEAT experiment. What they share is not the
science - AUROC and patient grouping belong to the medical benchmark, accuracy
belongs to the digit and object benchmarks - but the *execution policy*: the
smoke/pilot/full mode ladder, the command-line surface, the environment digest
that pins a run to an exact implementation, and the immutable v2 lock a full
result series runs under.

That policy lives here so it is written once and cannot drift between families.
Everything dataset-specific - manifests, audits, metrics, thresholds - stays in
each family's own package and is injected into these helpers rather than known
by them.
"""
