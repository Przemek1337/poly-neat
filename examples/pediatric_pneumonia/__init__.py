"""Pediatric chest X-ray pneumonia benchmark (NORMAL vs PNEUMONIA).

This package holds everything specific to the *Pediatric Pneumonia Chest X-ray*
dataset: the file-naming conventions of the archive, its integrity audit, the
frozen split manifest, the protocol lock and the experiment entry points that
assemble library components into the benchmark protocol.

The library itself stays dataset-agnostic. General image preprocessing,
training, metric and threshold machinery live under ``polyneat/``; only the
choice of X-ray-specific parameters and the conventions of this particular
archive live here.

The benchmark evaluates two tracks for every completed search seed:

* **Track A** - the exact checkpoint the search selected on
  ``search_validation``, with its trained weights and fitted preprocessing.
* **Track B** - the same topology with fresh weights, retrained on
  ``train + search_validation`` under one shared recipe.

Results describe this dataset and this budget. They are not a clinical
validation and must not be described as a reproduction of the DeepNEAT or
EXACT publications.

References:
    Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical
        Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*,
        172(5), 1122-1131. DOI: 10.1016/j.cell.2018.02.010
"""
