# Estimating Power without Measuring it: a Machine Learning Approach

G. Lemaitre and C. Lemaitre

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1202440.svg)](https://doi.org/10.5281/zenodo.1202440)

## Abstract

You can find the abstract linked to the experiments in the `abstract.pdf`. It
has been submitted to the Science & Cycling conference.

## Software

### Requirements

To run the script, you need to install
[`scikit-cycling`](https://github.com/scikit-cycling/scikit-cycling).

### Description

Two python files are available in this repository:

* `machine_learning_model.py`: This is our contribution for the current
  paper. We learn a gradient boosting regressor using heterogeneous data.
* `mathematical_model.py`: This is the model used as comparison to the machine
  learning model.

### Reproduce the experiments

``` bash
# Download the data
python download_data.py

# Run the two different models
python machine_learning_model.py
python mathematical_model.py
```
