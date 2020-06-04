# Bayesian Approach to Experimental Design Applied to Compton Scattering

<img align="right" width="140" src="./logos/buqeye_logo_web.png">
This repository contains the data and Jupyter notebooks to reproduce and extend the figures
in:

* Melendez, Furnstahl, Griesshammer, McGovern, Phillips, _Designing Optimal Experiments: An Application to Proton Compton Scattering_, [arXiv:2004.11307](https://arxiv.org/abs/2004.11307).

## Overview

The directory `notebooks` contains all the relevant Jupyter notebooks, including
the main notebooks `main_manuscript_analysis.ipynb`, `plot_compton_coefficients.ipynb`, 
and `order_exponent_analysis.ipynb`, which generate the figures in the
paper and saves them to the subdirectory `manuscript_figures`.
The directory `compton` contains the Python implementation code.
The raw data can be found in `data`. More information can be found in the 
annotated notebooks.

## Requirements and Installations

Due to the large size of the data files,
cloning this repo requires `git-lfs`.
The installation instructions for `git-lfs` can be found at
[git-lfs.github.com](https://git-lfs.github.com).

`Python 3` is
required with the (standard) packages listed in `requirements.txt` installed.
They can be installed by running the command:
``` shell
pip3 install -r requirements.txt
```
In addition, J. Melendez's package `gsum`, which is publicly available
[here](gsum) including installation instructions, needs to be installed
separately. Do not use `gsum` as installed by `pip3`.

With these prerequisites, to install this repository simply run (at the top
level):
```shell
pip3 install -e .
```

## Contact

To report any issues please use the issue tracker.

## Citing this Work and Further Reading

* Melendez, Furnstahl, Griesshammer, McGovern, Phillips, _Designing Optimal Experiments: An Application to Proton Compton Scattering_, [arXiv:2004.11307](https://arxiv.org/abs/2004.11307).



[buqeye]:https://buqeye.github.io/ "to the website of the BUQEYE collaboration"
[gsum]:https://github.com/buqeye/gsum "to the gsum github repository"

