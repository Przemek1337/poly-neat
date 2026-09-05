# DeepNEAT in PolyNEAT

This package implements DeepNEAT: a chromosome is a DAG
whose nodes are trainable layers, whose edges are unweighted tensor flows, and
whose fitness is validation accuracy after training the decoded network from a
fresh initialization. It is DeepNEAT, not CoDeepNEAT: modules and blueprints are
not co-evolved.

Primary source for the detailed DeepNEAT algorithm and CIFAR-10 experiment:

- J. Liang, [Evolution of Deep Neural Network Architectures, Chapter 3](https://nn.cs.utexas.edu/downloads/papers/liang.thesis2018.pdf), 2018.

The related R. Miikkulainen et al.
[Evolving Deep Neural Networks](https://arxiv.org/abs/1703.00548) paper focuses
primarily on CoDeepNEAT. PolyNEAT's `deepneat_paper` profile name is retained
for backwards compatibility, but its 8.9% reference result and Table 3.1 search
space come from Liang's DeepNEAT chapter, not the paper's CoDeepNEAT result.

## Source alignment and remaining differences

The structural core follows DeepNEAT: minimal initial graphs, incremental
node/edge mutations, historical markings, crossover, speciation, layer-level
genes, unweighted edges, arbitrary skip connections, and fixed-duration
training for fitness.

The implementation carries the source experiment's node-level and global genes
and uses the mutation semantics described there. The table distinguishes the
general method from the concrete CIFAR-10 search space; examples such as
recurrent layers and evolvable activations are possibilities named by the
general method, not genes used in Table 3.1's image experiment.

| Area | Source DeepNEAT | PolyNEAT |
|---|---|---|
| Layer types | domain-specific; CIFAR-10 nodes are convolutional | CIFAR-10 profile inserts convolutional nodes; dense is an additional supported domain option |
| Node mutation | Gaussian perturbation of real values and bit flips | same; categorical kernel/type genes switch to another allowed value |
| Activation | may be a node property generally; not evolved in CIFAR-10 Table 3.1 | fixed ReLU in the CIFAR-10 implementation |
| Weight initialization | evolved initial-weight scale | same scalar gene applied after every fresh PyTorch initialization |
| Global hyperparameters | learning rate, momentum, Nesterov, HSV/crop/distortion/flip/normalization | same chromosome-wide genes and Table 3.1 ranges |
| Merge | concat or sum; image parents downsampled with max pooling to the smallest output size | concat; adaptive max pooling to the smallest spatial shape before conv, dense, or output layers |
| Evaluation distribution | many GPU workers | sequential on one device |

The source does not specify every low-level operator detail and no official
DeepNEAT reference implementation is linked from it. This is therefore a
source-aligned reimplementation, not a bit-for-bit copy. In particular, the
source gives no mutation probabilities, Gaussian standard deviation, exact
speciation distance, batch size, random seed, base initializer, or choice
between concat and sum. Those are explicit implementation parameters. The
source profile disables PolyNEAT's invented layer-hyperparameter distance
(`c3 = 0`) and parameter-count fitness cutoff; the smoke profile may enable
safety extensions for practical execution.

## Dataset integrity and reproducibility

MNIST, Fashion-MNIST and CIFAR-10 preserve their official train/test split.
Validation is carved only from official train. Variance normalization is an
evolved boolean gene; when enabled, its statistics are fitted only on the
fitness-training rows and reused for validation and test. Neither validation
nor test contributes statistics.

Before training each non-degenerate phenotype, the evaluator derives a seed
from the run seed and generation, resets every Linear/Conv/BatchNorm module,
and then constructs SGD with the chromosome's evolved learning rate, momentum
and Nesterov setting. All candidates in one generation receive common
random numbers, so identical genomes have identical initialization and batch
order regardless of population position. Set
`use_deterministic_training_algorithms: true` for deterministic PyTorch kernels
where supported, with the expected throughput cost.

## Was DeepNEAT evaluated on Fashion-MNIST?

No. The source DeepNEAT image-classification experiment used **CIFAR-10**.
Fashion-MNIST and MNIST in this repository are additional, cheaper benchmarks;
they cannot be presented as reproductions of the published experiment.

## Source-budget CIFAR-10 protocol

`examples/cifar10/deepneat_paper.py` and `deepneat_paper.yaml` reproduce the
source experiment's dataset and compute budget:

| Protocol item | Source | `cifar10/deepneat_paper` |
|---|---:|---:|
| Dataset | CIFAR-10 official 50k/10k | same |
| Fitness train/validation | 42,500 / 7,500 | same |
| Population | 100 | 100 |
| Evolution limit | 60 generations | 60 |
| Training per fitness | 8 epochs | 8 |
| Final training | all 50,000 rows, 300 epochs | same |
| Test use | final report only | final report only |
| Kernel sizes | `{1, 3}` | `{1, 3}` |
| Dropout range | `[0.0, 0.7]` | `[0.0, 0.7]` |
| Filter range | integer range `[32, 256]` | same |
| Initial weight scale | `[0.0, 2.0]` | same |
| Learning rate | `[0.0001, 0.1]` | same, evolved |
| Momentum | `[0.68, 0.99]` | same, evolved |
| Crop / spatial scaling | `[26, 32]` / `[0.0, 0.3]` | same, evolved |
| HSV shift/scale, flips, variance normalization, Nesterov | evolved | same |

The source reports **8.9% CIFAR-10 test error (91.1% accuracy)** for its best
DeepNEAT network. That is the historical reference point, not an expectation
of equality: framework, initialization base distribution, merge choice,
parallelism and source-unspecified operator details can all affect results.

At the end of a run, the example prints `test_error` and
`test_error_gap_to_paper_percentage_points`. A positive gap means PolyNEAT's
error is higher (worse) than the dissertation's 8.9%; a negative gap means it
is lower. The metric retains its legacy name for output compatibility. This is
a real held-out comparison because the official test set is not used for
evolution, preprocessing statistics or architecture selection.

## Single-GPU smoke/reference profile

Run this profile first on a single L40:

```bash
uv run python -m examples.cifar10.deepneat_smoke --gpu --no-tensorboard
```

It is designed to target less than two hours on an otherwise idle
[NVIDIA L4](https://www.nvidia.com/en-au/data-center/l4/) 24 GB (the server
target declared in this repository), although this cannot be guaranteed before
measuring the particular server and the architectures produced by evolution.
NVIDIA's physical [L40](https://www.nvidia.com/en-us/data-center/l40/) has 48 GB;
if `nvidia-smi` shows 24 GB for an L40, it may be a partitioned/vGPU environment
with different performance characteristics.

| Budget item | Smoke | Paper profile |
|---|---:|---:|
| Fitness evaluations | 12 x 8 = 96 | 100 x 60 = 6,000 |
| Fitness train/validation | 8,500 / 1,500 | 42,500 / 7,500 |
| Epochs per fitness | 2 | 8 |
| Final training | 10,000 rows, 30 epochs | 50,000 rows, 300 epochs |
| Official test | all 10,000 rows | all 10,000 rows |
| Parameter cap | 5 million (safety extension) | disabled (not reported by source) |

The first smoke result is the baseline for later code changes. Compare runs
using the same YAML, seed and hardware: `test_accuracy` says which run is
better, while `runtime_seconds` covers the entire command, including final
training. The printed gap to the
dissertation's 8.9% error is useful context, but it is **not** evidence that the
algorithm is better or worse than published DeepNEAT because the search and
training budgets are deliberately much smaller.

## Running it on a Linux CUDA server

The locked Linux environment installs the CUDA 12.6 PyTorch build. From a
fresh clone, verify the driver and create the environment:

```bash
nvidia-smi
uv sync --frozen
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The final value in the diagnostic command must be a CUDA device name, and
`torch.cuda.is_available()` must be `True`. Then run it in `tmux`, so an SSH
disconnect does not stop the process:

```bash
tmux new -s deepneat
uv run python -m examples.cifar10.deepneat_smoke --gpu --no-tensorboard 2>&1 | tee deepneat-cifar10-smoke.log
```

Detach with `Ctrl-b`, then `d`; return with `tmux attach -t deepneat`. The first
run downloads the official CIFAR-10 Python archive. The console log contains
the final paper gap, while genomes and TensorBoard events are written to
`examples/cifar10/artifacts/deepneat_paper/`. To inspect TensorBoard through an
SSH tunnel after or during the run:

```bash
uv run tensorboard --logdir examples/cifar10/artifacts/deepneat_paper/tensorboard --host 127.0.0.1 --port 6006
# on the local machine: ssh -L 6006:127.0.0.1:6006 USER@SERVER
```

After the smoke run, the full paper-budget profile can be started with:

```bash
uv run python -m examples.cifar10.deepneat_paper --gpu --no-tensorboard 2>&1 | tee deepneat-cifar10-paper.log
```

The full configuration evaluates 6,000 candidate networks (100 candidates x
60 generations), each for eight epochs, and then trains the winner for 300
epochs. The source distributed each generation over 100 GPU-equipped EC2
workers; this repository currently evaluates candidates sequentially on one
device. Therefore a single-GPU run is protocol-comparable but can take a very
long time and is not runtime-comparable to the published experiment.

For a cheap wiring check that does not download data or execute the full
budget:

```bash
uv run pytest -q tests/test_cifar10_dataset.py tests/test_deepneat_examples.py
```
