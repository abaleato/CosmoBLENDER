# CosmoBLENDER

**C**osmological **B**iases to **LEN**sing and **D**elensing **D**ue to **E**xtragalactic **R**adiation

CosmoBLENDER is a python code that computes biases to CMB lensing auto- and cross-correlations, as well as internal
delensing. It follows [Baleato Lizancos et al. 2025 (in prep.)]().

## Installation
###### Dependencies:
- `NumPy`, `SciPy`, `Matplotlib`
- `BasicILC` from [this fork](https://github.com/abaleato/BasicILC/tree/cosmoblender)
- `Hmvec` from its [galaxy branch](https://github.com/simonsobs/hmvec) for CIB calculations
- `Quicklens` ([Python 3 version](https://github.com/abaleato/Quicklens-with-fixes/tree/Python3))
- `astropy`
- [`pyccl`](https://github.com/LSSTDESC/CCL) (optional, needed for FFTlog-based calculation; we recommend using the default Gaussian quadratures instead).


###### Editable installation in-place:
First, clone the repository

    git clone https://github.com/abaleato/CosmoBLENDER.git

and from within it run

    python -m pip install -e .

## Usage
See the notebooks under the `examples/` directory -- `examples/example_pipeline.ipynb` is a great place to start.

## Attribution
If you use the code, please cite [Baleato Lizancos et al. 2025 (in prep.)]().
