# MNIST benchmark — server runbook

## Compatibility after the protocol fixes

Repeat pilots and create new locks after updating: earlier MNIST runs lost gray
levels during conversion and cannot be used as results for this corrected protocol.
Old checkpoints/locks are intentionally incompatible with the corrected code and
content-based data identity. Preserve them separately; use new artifact directories.

Only profiles declaring `protocol.profile_kind: smoke` default to smoke. Other
profiles default to pilot; an explicit `--mode smoke` on a result profile is rejected.
Completed `--resume` runs validate data, configuration, environment and lock, then
return the saved report without retraining or rescoring. Interrupted track B may
still restart. Search-resume equivalence is tested under controlled CPU conditions,
not guaranteed bit-for-bit across GPU hardware.

The data fingerprint includes pixels, labels and split order; provenance names
are separate, so confirming the release/license during freeze does not alter it.

The DeepNEAT vs EXACT comparison on MNIST, run on the GPU box in five stages:
**smoke → pilot → freeze → seed series → resume**. MNIST is single-channel, the
input EXACT was designed for, so neither algorithm is bent to fit; the color
task lives in the separate CIFAR experiment.

MNIST downloads itself, so there is no archive to point at. Every command takes
`--gpu` (it errors rather than silently falling back to CPU) and writes a
`run_report.json` recording the exact configuration, environment and result.

Two runnable methods, distinct from the reproduction runners
`examples.mnist.deepneat` / `examples.mnist.exact`:

```
python -m examples.mnist.benchmark_deepneat
python -m examples.mnist.benchmark_exact
```

## 1. Smoke — prove the plumbing (minutes, either device)

```bash
uv run python -m examples.mnist.benchmark_deepneat --cpu
uv run python -m examples.mnist.benchmark_exact --cpu
```

A tiny population over a few hundred digits, exercising both tracks, the
checkpoint round-trip and the held-out test read. Its numbers mean nothing; a
non-zero exit or a traceback is the only thing to read here.

## 2. Pilot — measure cost without touching the test (GPU)

The pilot runs the full profile on real data but **never scores the official
test** — that is what lets its report justify a freeze. Give each method its own
artifacts directory.

```bash
uv run python -m examples.mnist.benchmark_deepneat --gpu --mode pilot \
    --config examples/mnist/configs/deepneat_full.yaml \
    --artifacts-directory results/mnist/pilot/deepneat
uv run python -m examples.mnist.benchmark_exact --gpu --mode pilot \
    --config examples/mnist/configs/exact_full.yaml \
    --artifacts-directory results/mnist/pilot/exact
```

Read `results/mnist/pilot/<method>/run_report.json`: `official_test` must be
`{}`, and the search runtime tells you whether `search_budget_seconds` in the
full profiles is right. Adjust the budget in both full profiles if needed and
**re-run the pilot** — the freeze checks that the pilot ran the exact profile it
will freeze.

## 3. Freeze — turn the pilots into an immutable lock (no test)

```bash
uv run python -m examples.mnist._freeze \
    --output-directory results/mnist/series \
    --seeds 101 102 103 104 105 \
    --protocol-id mnist-benchmark-full-v1 \
    --dataset-release mnist/official \
    --dataset-license mnist-official-yann-lecun-terms \
    --pilot-reports results/mnist/pilot/deepneat/run_report.json \
                    results/mnist/pilot/exact/run_report.json \
    --gpu
```

This writes `results/mnist/series/protocol.lock.yaml`. It refuses to freeze if a
pilot scored the test, ran a different profile, used a different environment, or
if the provenance still reads like a placeholder. The lock is self-checksummed:
editing it afterwards makes every full run refuse to start. Changing anything
after this point starts a new study, not an amendment.

## 4. Seed series — the actual comparison (GPU)

One directory per method and seed. The full mode is bound to the lock and errors
on any mismatch.

```bash
for method in deepneat exact; do
  for seed in 101 102 103 104 105; do
    uv run python -m examples.mnist.benchmark_${method} --gpu --mode full \
        --config results/mnist/series/profiles/${method}.yaml \
        --protocol-lock results/mnist/series/protocol.lock.yaml \
        --seed ${seed} \
        --artifacts-directory results/mnist/series/${method}/seed_${seed}
  done
done
```

Each run reports track A (the network the search kept) and track B (that same
topology retrained from fresh weights under the one shared recipe), their test
accuracy and error, the parameter count, the wall-clock time and the fraction of
failed evaluations.

## 5. Resume — after an interruption

A search checkpoints at every completed evaluation and every generation
boundary. To continue one that stopped, re-run the same command with `--resume`.
After an **unclean** stop (the process was killed), the time between the last
checkpoint and the stop is not free: read it from the scheduler or the logs and
charge it back so the budget stays honest.

```bash
uv run python -m examples.mnist.benchmark_deepneat --gpu --mode full \
    --config results/mnist/series/profiles/deepneat.yaml \
    --protocol-lock results/mnist/series/protocol.lock.yaml \
    --seed 101 \
    --artifacts-directory results/mnist/series/deepneat/seed_101 \
    --resume --lost-work-seconds 240
```

Under controlled CPU conditions, the tests verify that interrupted searches
select the same model as uninterrupted ones for both algorithms. This is not
a guarantee of bit-for-bit equivalence across GPU hardware.

## What this comparison is and is not

- **Not** a reproduction of either source experiment. Those are the separate
  `examples.mnist.deepneat` / `examples.mnist.exact` runners, left untouched.
- The two methods learn in different places — DeepNEAT trains each candidate
  from scratch, EXACT inherits and refines weights between generations — so the
  shared wall-clock budget buys them different numbers of evaluations. That is
  reported (evaluations, time, model size) rather than hidden.
- The budgets in the full profiles are pilot outputs. A series run before the
  pilot has set them is a pilot run whose numbers do not belong in a table.
```
