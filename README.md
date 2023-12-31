[![DOI](https://zenodo.org/badge/715070348.svg)](https://zenodo.org/doi/10.5281/zenodo.10091434)

# Exponential adoption of battery electric cars

This contains the code necessary to reproduce the figures in the article **Exponential adoption of battery electric cars** by Felix Jung, Malte Schröder and Marc Timme.

**Important note**: For copyright reasons, this repository contains no actual data. To reproduce the plots from the paper, you must first acquire the actual data from the various sources provided in the manuscript.

## Setup

### Prerequisites

- git
- Python 3.10 with `pip`
- [pdf2ps](https://linux.die.net/man/1/pdf2ps)
- [ps2eps](https://linux.die.net/man/1/ps2eps)

### Setup

```bash
git clone https://github.com/PhysicsOfMobility/exponential-adoption-of-battery-electric-cars.git
cd exponential-adoption-of-battery-electric-cars
pip install -e .
```

### Usage

Run
```bash
jupyter lab
```

Then, in JupyterLab, navigate to the cloned repository, enter the "notebooks" folder, right-click on `manuscript_figures.py` and select "Open With > Notebook". Then, run and explore the notebook.
