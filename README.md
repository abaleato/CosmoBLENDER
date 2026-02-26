# CosmoBLENDER

[![PyPI version](https://img.shields.io/pypi/v/cosmoblender)](https://pypi.org/project/cosmoblender/)
[![Python](https://img.shields.io/pypi/pyversions/cosmoblender)](https://pypi.org/project/cosmoblender/)
[![License: CC BY-NC-SA 3.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/3.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.03859-b31b1b.svg)](https://arxiv.org/abs/2507.03859)

**C**osmological **B**iases to **LEN**sing and **D**elensing **D**ue to **E**xtragalactic **R**adiation

CosmoBLENDER computes foreground-induced biases to temperature-based CMB lensing reconstructions, including:
- CMB lensing auto-correlations,
- CMB lensing cross-correlations with galaxies,
- internal B-mode delensing.

The implementation follows [Baleato Lizancos et al. 2025](https://arxiv.org/abs/2507.03859).

## Installation

### From PyPI

    pip install cosmoblender

### From source (editable)
Clone the repository:

    git clone https://github.com/abaleato/CosmoBLENDER.git

From the repository root, run:

    pip install -e .

### Dependencies
- `NumPy`, `SciPy`, `Matplotlib`, `astropy`
- `BasicILC` from [this fork](https://github.com/abaleato/BasicILC/tree/cosmoblender)
- `Hmvec` from its [galaxy branch](https://github.com/simonsobs/hmvec) for CIB calculations
- `Quicklens` ([Python 3 version](https://github.com/abaleato/Quicklens-with-fixes/tree/Python3))
- [`pyccl`](https://github.com/LSSTDESC/CCL) (optional, for FFTlog-based calculations)

## Notebooks
All tutorials and worked examples are in `notebooks/`.

Start with `notebooks/example_pipeline.ipynb`.

- `notebooks/example_pipeline.ipynb`: An introduction to data stuctures, experiment setup and bias calculations for auto-correlations, cross-correlations, and delensing.
- `notebooks/auto_correlations.ipynb`: Examples focused on biases to the CMB lensing auto-spectrum, including how to vary cosmological, astrophysical and experimental parameters.
- `notebooks/cross_correlations.ipynb`: Same as above, but focusing on cross-correlations with galaxy samples.
- `notebooks/delensing.ipynb`: Examples focused on foreground-induced biases in B-mode delensing.
- `notebooks/varying_M_min.ipynb`: Studies how predictions change as the minimum halo mass cut is varied.
- `notebooks/varying_z.ipynb`: Same as above, but for the redshift range.
- `notebooks/explore_zerocrossing_primbispec.ipynb`: Investigates where and why primary bispectrum contributions cross zero.

## Attribution
If you use the code, please cite [Baleato Lizancos et al. 2025](https://arxiv.org/abs/2507.03859).
