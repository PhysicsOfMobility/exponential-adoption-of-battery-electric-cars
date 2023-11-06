# Exponential adoption of battery electric cars

This contains the code necessary to reproduce the figures in the article **Exponential adoption of battery electric cars** by Felix Jung, Malte Schröder and Marc Timme.

**Important note**: For copyright reasons, this repository contains only dummy data. To reproduce the actual plots from the paper, you must first acquire the actual data from the various sources provided in the manuscript.

## Setup

### Prerequisites

- [pdf2ps](https://linux.die.net/man/1/pdf2ps)
- [ps2eps](https://linux.die.net/man/1/ps2eps)
- Python 3.10 with `pip`
- git

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
