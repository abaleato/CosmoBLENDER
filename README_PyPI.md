# CosmoBLENDER

**C**osmological **B**iases to **LEN**sing and **D**elensing **D**ue to **E**xtragalactic **R**adiation

CosmoBLENDER computes foreground-induced biases to temperature-based CMB lensing reconstructions, including:
- CMB lensing auto-correlations,
- CMB lensing cross-correlations with galaxies,
- internal B-mode delensing.

The implementation follows [Baleato Lizancos et al. 2025](https://arxiv.org/abs/2507.03859).


## Installation
Install from PyPI with:

     python -m pip install cosmoblender

Optional FFTlog functionality requires `pyccl`:

     python -m pip install "cosmoblender[fftlog]"

### Core dependencies
- `NumPy`, `SciPy`, `Matplotlib`
- `BasicILC` from [this fork](https://github.com/abaleato/BasicILC/tree/cosmoblender)
- `Hmvec` from its [galaxy branch](https://github.com/simonsobs/hmvec) for CIB calculations
- `Quicklens` ([Python 3 version](https://github.com/abaleato/Quicklens-with-fixes/tree/Python3))
- `astropy`

## Usage
Tutorial notebooks live in `notebooks/` in the GitHub repository.

Start with `notebooks/example_pipeline.ipynb`, then see:
- `notebooks/auto_correlations.ipynb`
- `notebooks/cross_correlations.ipynb`
- `notebooks/delensing.ipynb`

## Attribution
If you use the code, please cite [Baleato Lizancos et al. 2025](https://arxiv.org/abs/2507.03859).