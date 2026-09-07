# Pediatric pneumonia chest X-ray benchmark

Classification of pediatric chest radiographs into `NORMAL -> 0` and
`PNEUMONIA -> 1`, used to compare DeepNEAT and EXACT under one budget, against
two mandatory controls and one separately-reported transfer-learning baseline.

Results from this package describe **this dataset and this budget**. They are
not a clinical validation, not evidence of performance at another site, and not
a reproduction of the DeepNEAT or EXACT publications.

## Getting the data

The archive is not in the repository and never will be: no images, no cache, no
checkpoints, no predictions. Download it yourself with the Kaggle CLI:

```bash
kaggle datasets download -d andrewmvd/pediatric-pneumonia-chest-xray
unzip pediatric-pneumonia-chest-xray.zip -d examples/pediatric_pneumonia/data/kaggle
```

Record the release you actually got. The dataset derives from Kermany et al.
(2018); the source Mendeley Data v2 release states CC BY 4.0, and the manifest
must name the Kaggle release used and its stated license, not an assumed one.

### Expected tree

Any one of these nesting levels is accepted, and exactly one of them may be
present. Two competing roots is an error, not permission to load the data
twice:

```
<data-dir>/train|val|test/{NORMAL,PNEUMONIA}/*.jpeg
<data-dir>/chest_xray/train|val|test/{NORMAL,PNEUMONIA}/*.jpeg
<data-dir>/chest_xray/chest_xray/train|val|test/{NORMAL,PNEUMONIA}/*.jpeg
```

The expected counts are 5,856 images: 1,349 NORMAL and 3,883 PNEUMONIA in the
original `train+val`, and 234 NORMAL and 390 PNEUMONIA in the official `test`.
These are a cross-check against the download, not proof of integrity or of
patient disjointness. Any deviation is reported and the manifest records what
was actually found.

### How this package is laid out

```
configs/          one yaml per profile, plus protocol.lock.template.yaml
<profile>.py      the five runnable entry points, one per method
dataset.py        decoding, resizing and the split cache
_*.py             audit, manifest, protocol lock, stage controller, methods
```

A profile reads the yaml named after it, so `deepneat_smoke.py` reads
`configs/deepneat_smoke.yaml`. The pairing is derived from the module name in
`config_path_for`, not written out per profile, so a rename cannot leave the
two pointing at different files.

### Pointing a run at your copy

Every profile takes its data directory from its yaml. The smoke profiles
default to a synthetic archive generated on first use, so they run with no
download at all:

```bash
uv run python -m examples.pediatric_pneumonia.deepneat_smoke --cpu
uv run python -m examples.pediatric_pneumonia.exact_smoke --cpu
uv run python -m examples.pediatric_pneumonia.random_search_smoke --cpu
uv run python -m examples.pediatric_pneumonia.fixed_cnn_smoke --cpu
uv run python -m examples.pediatric_pneumonia.transfer_learning_smoke --cpu
```

The synthetic images are not radiographs. A smoke run checks the plumbing -
both tracks, threshold selection, checkpoint round-trip, prediction export,
the bootstrap - and its numbers must never appear in a results table.

### Choosing the device

`--cpu` and `--gpu` are mutually exclusive and pick the one device the whole
run uses: the search, both training tracks, threshold scoring and the test
evaluation. There is no per-stage device and no silent fallback - `--gpu`
without CUDA exits rather than quietly producing CPU timings, which under a
wall-clock budget would change what the budget actually bought. Passing
neither keeps the CPU, which is what the smoke profiles are sized for; the
pilot and the result series are meant for `--gpu`.

The device that was actually used is written into `effective_configuration`
in every run report, so a results table can be checked against it rather than
trusted.

## Installing what the benchmark needs

Image decoding, the reference metric implementations and the transfer-learning
backbone live in an optional extra, pinned together:

```bash
uv sync --extra benchmark
```

The pin is deliberate. AUROC and average precision come from scikit-learn, and
a metric implementation that changed mid-series would make its numbers
incomparable; torchvision ships compiled operators against one torch ABI, and
installing it unpinned upgrades torch underneath and breaks every torchvision
call. Each run also records the versions it actually imported.

## What the protocol does

1. **Audit.** Walk the archive once: resolve the root, decode every image,
   hash the file and the decoded pixels, parse the grouping key from the file
   name, and record every problem found. Conflicting labels, a duplicate shared
   between the development pool and the official test set, and a confirmed
   patient on both sides are blocking. Perceptual similarity is only ever a
   warning: it is a prompt to review, never proof of a duplicate and never
   grounds for deleting an image.
2. **Freeze the split.** The original train and val become one development
   pool, split once by group into 70/15/15. Group disjointness outranks the
   exact proportions, and the realised counts are recorded rather than assumed.
   The official test directory is carried over untouched.
3. **Search.** Each method sees train and search_validation only. Fitness is
   always AUROC on search_validation for the PNEUMONIA class.
4. **Track A.** The exact checkpoint the search selected, with the weights that
   earned that fitness. For DeepNEAT this cannot be rebuilt from the genome -
   its genome carries no weights at all - so the snapshot taken during
   evaluation is what gets frozen, and it is re-scored to confirm it reproduces
   the selected number.
5. **Track B.** The same topology, fresh parameters, retrained on
   train+search_validation under one shared recipe with new statistics and new
   class weights. EXACT genomes are reset first: clearing the kernels alone
   leaves the evolved batch-normalization state behind, and an already-trained
   genome is skipped by its trainer entirely.
6. **Threshold.** Each frozen checkpoint gets its own threshold, chosen on
   threshold_validation by Youden's J and bound to the digests of that
   checkpoint, its preprocessing and the manifest. Thresholds are never carried
   between tracks or seeds.
7. **Test.** Predictions, metrics at the chosen threshold and at the reference
   threshold of 0.5, and bootstrap intervals - computed once, after every model
   and threshold is already fixed.

## What a run leaves behind

```
<artifacts>/manifest.json                    the frozen split and its digest
<artifacts>/run_report.json                  statuses, thresholds, test metrics,
                                             effective configuration
<artifacts>/checkpoints/<model>.pt           weights + preprocessing state
<artifacts>/predictions/<model>_<split>.json logits, probabilities, example ids
```

Benchmark series give every seed its own directory:

```bash
uv run python -m benchmarks.run_benchmark pediatric_pneumonia/deepneat_smoke \
    --repeats 3 --base-seed 101 --artifacts-root benchmarks/results/pneumonia --cpu
```

## Limitations this benchmark states rather than hides

- **Patient independence is not established.** The shipped train and test
  directories number their patients independently, so `person1` in one is not
  the person behind `person1` in the other. Identifiers are therefore compared
  only within a pool, and file names can neither confirm nor rule out that a
  patient appears on both sides. The manifest records this, and the test
  bootstrap resamples images rather than patients when it holds.
- **One split.** Every method and seed shares one manifest, so the evaluation
  is conditional on that split. Repeating seeds is not a substitute for
  validating on an independent dataset.
- **Different search spaces.** DeepNEAT and EXACT evolve different kinds of
  network. The difference in results cannot be attributed to their evolutionary
  operators alone.
- **Random search is not identically distributed.** It shares DeepNEAT's
  decoder, genes, constraints, fitness, budget and candidate training, and
  differs in having no selection pressure. Its structural-mutation range is a
  stated sampling distribution, not the distribution DeepNEAT's population
  reaches after selection.
- **Transfer learning is not budget-comparable.** The cost of its ImageNet
  pretraining is counted nowhere here, so it is reported separately.
- **Intervals are conditional.** A bootstrap interval covers the sampling of
  the test set given a fixed model and threshold. It does not cover the
  uncertainty of the threshold search or the randomness of the whole
  architecture search; seed-to-seed spread is reported separately.

## Sources

- Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical
  Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*,
  172(5), 1122-1131. https://doi.org/10.1016/j.cell.2018.02.010
- Kermany, Zhang, Goldbaum (2018). Labeled Optical Coherence Tomography (OCT)
  and Chest X-Ray Images for Classification. Mendeley Data v2.
- Miikkulainen, R., et al. (2017). Evolving Deep Neural Networks.
  arXiv:1703.00548.
- Desell, T. (2017). Developing a Volunteer Computing Project to Evolve
  Convolutional Neural Networks and Their Hyperparameters. IEEE e-Science.
- Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1),
  32-35.
