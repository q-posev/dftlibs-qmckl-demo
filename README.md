# dftlibs-qmckl-demo

This repository contains Jupyter notebooks (in Markdown `.md` format),
which demonstrate the use of the open-source libraries like 
`qmckl`, `numgrid`, `trexio` and `xcfun` via their Python APIs. 
It takes advantage of the high-performance optimization of the underlying libraries 
(C, C++, Rust) and their native compatibility with Python. 


## Contents

- `qmckl_numgrid.md`
  1. Load some quantum chemistry data including molecular geometry and basis set information stored in the [trexio](https://github.com/TREX-CoE/trexio) file format via the `qmckl` context
  2. Set up a grid for molecular integrals via the `numgrid` library using the pre-processed data from the `qmckl` context
  3. Compute atomic orbital (AO) values on a grid via the high-performance `qmckl` routines
  4. Calculate the AO overlap matrix (via the `qmckl_dgemm` or NumPy matrix multiplication routines)
  5. Compute density matrix using the molecular orbital (MO) coefficients from the context
  6. Calculate numerically the number of electrons and compare it with the exact value from the context
  7. Compute density values on a grid (required for the DFT)
  8. Set up the exchange-correlation functional via the `xcfun` library
  9. Compute energy and XC potential values on a grid (via the `xcfun`)
  10. Calculate numerically the XC energy contribution


## Requirements

- `python3` 	(>= 3.6) 
- `jupyter`
- `jupytext` 	(to convert notebooks from the Markdown `.md` format into`.ipynb` notebook)

- `numpy`	(>= 1.19.3)
- `trexio`	(Python API installation: `pip install trexio`)
- `numgrid` 	(Python API installation: `pip install numgrid`) 
- `qmckl` 	([Python API installation instructions](https://github.com/TREX-CoE/qmckl/blob/master/README.md))
- `xcfun`	([Python API installation instructions](https://xcfun.readthedocs.io/en/latest/building.html#))

**Note:** we recommend to use virtual environments to avoid compatibility issues and to improve reproducibility.
For more details, see the corresponding part of the [Python documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments).


## Running the notebook

To obtain a local copy of the `.ipynb` files: 

- Clone the repository
- `jupytext --to notebook qmckl_numgrid.md`
- `jupyter notebook qmckl_numgrid.ipynb`


### Additional steps needed to run a custom virtual environment in Jupyter notebooks

In some cases, it may happen that the Jupyter kernels in the activated virtual environment 
(e.g. `myvenv`) still point to the system-wide python binaries and not to the environment ones.
This will result in `ImportError` when importing custom packages in the notebook cell. 
In order to avoid this, the `myvenv` has to be installed as an additional kernel.

This requires `ipykernel` python package, which usually comes together with the Jupyter installation. If this is not the case, run `pip install ipykernel`.
You can install `myvenv` as a kernel by executing the following command:

`python3 -m ipykernel install --user --name=myvenv`

Now you can launch a Jupyter notebook. Once it is open, make sure that your virtual environment is selected as the current kernel.
If this is not the case, try this:

1. Press the `Kernel` button in the navigation panel
2. In the output list of options select `Change kernel`
3. Find the name of your virtual environment (e.g. `myvenv`) in the list and select it

To uninstall the kernel named `myvenv`, execute the following command:

`jupyter kernelspec uninstall myvenv`
