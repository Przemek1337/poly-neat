# CIFAR-10 benchmark — server runbook

After the protocol fixes, repeat pilots and create new locks in new directories.
Old checkpoints/locks are incompatible with the corrected code and content-based
dataset identity. Only profiles declaring `protocol.profile_kind: smoke` may run
in smoke mode; result profiles default to pilot when no mode is given.

Completed `--resume` runs validate data, configuration, environment and lock and
return the saved report without retraining or rescoring. Dataset fingerprints
cover pixels, labels and split order, independently of release/license labels.

The **additional colour experiment**: DeepNEAT on CIFAR-10, run on the GPU box
in the same stages as the MNIST comparison — **smoke → pilot → freeze → seed
series → resume**. CIFAR runs **DeepNEAT only**: EXACT is single-channel by
construction, so the two-method comparison lives on MNIST, and the source
reproduction lives in `examples.cifar10.deepneat_paper`. This runner is separate
from both.

CIFAR is lower priority than MNIST and RTG and need not block them; run it when
compute time is free.

CIFAR-10 downloads itself on first use. Every command takes `--gpu` (it errors
rather than silently falling back to CPU) and writes a `run_report.json`.

```
python -m examples.cifar10.benchmark_deepneat
```

## 1. Smoke — prove the plumbing (minutes)

```bash
uv run python -m examples.cifar10.benchmark_deepneat --cpu
```

## 2. Pilot — measure cost without touching the test (GPU)

```bash
uv run python -m examples.cifar10.benchmark_deepneat --gpu --mode pilot \
    --config examples/cifar10/configs/deepneat_full.yaml \
    --artifacts-directory results/cifar10/pilot/deepneat
```

`official_test` in the report must be `{}`. Read the search runtime and adjust
`search_budget_seconds` in `deepneat_full.yaml` if needed, then re-run the pilot.

## 3. Freeze — turn the pilot into an immutable lock (no test)

```bash
uv run python -m examples.cifar10._freeze \
    --output-directory results/cifar10/series \
    --seeds 101 102 103 104 105 \
    --protocol-id cifar10-benchmark-full-v1 \
    --dataset-release cifar10/official \
    --dataset-license cifar10-krizhevsky-2009-mit \
    --pilot-reports results/cifar10/pilot/deepneat/run_report.json \
    --gpu
```

Writes `results/cifar10/series/protocol.lock.yaml`, self-checksummed and refused
if edited. Changing anything afterwards starts a new study.

## 4. Seed series (GPU)

```bash
for seed in 101 102 103 104 105; do
  uv run python -m examples.cifar10.benchmark_deepneat --gpu --mode full \
      --config results/cifar10/series/profiles/deepneat.yaml \
      --protocol-lock results/cifar10/series/protocol.lock.yaml \
      --seed ${seed} \
      --artifacts-directory results/cifar10/series/deepneat/seed_${seed}
done
```

Each run reports track A (the network the search kept) and track B (that same
topology retrained from fresh weights under the shared recipe), test accuracy and
error, parameter count, wall-clock time and the failed-evaluation fraction.

## 5. Resume

Re-run the same command with `--resume`; after an unclean stop add
`--lost-work-seconds <n>` from the scheduler or logs so the budget stays honest.
Search resume is tested for equivalence under controlled CPU conditions; this
does not promise bit-for-bit equivalence across GPU hardware. Interrupted track B
may restart.

## What this is and is not

- **Not** a reproduction of the source experiment — that is
  `examples.cifar10.deepneat_paper`, left untouched.
- CIFAR carries no EXACT: the two-method comparison is MNIST's, on the
  single-channel input EXACT was designed for. The budgets in the full profile
  are pilot outputs; a series run before the pilot has set them is a pilot run.
```
