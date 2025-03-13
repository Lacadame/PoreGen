poregen
==============================

Neural networks for porous media

--------

For a tutorial, see notebooks/tutorials/0001-basic-usage.ipynb.

--------
Along the development of this project, we've used extensively the following resources:

The Eleven Sandstones database at Digital Rocks Portal, found at [https://www.digitalrocksportal.org/projects/317](https://www.digitalrocksportal.org/projects/317), which is made available under the ODC Attribution License.

The Micro-CT Images and Networks, found at [https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/), kindly provided freely available by Professor Martin Blunt and the Pore-Scale Modelling and Imaging Group at Imperial College London.
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                     the creator's initials, and a short `-` delimited description, e.g.
    │   |                     `1.0-jqp-initial-data-exploration`.
    |   ├── exploratory    <- Notebooks just to tinker and train models.
    |   ├── tutorials      <- Tutorials for understanding the usage.
    |   
    |
    ├── poregen            <- Source code for use in this project.
    |   |
    │   ├── __init__.py    <- Makes poregen a Python module
    |   |
    |   ├── global_constants.py   <- All the global constants used in the project
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── toy_datasets.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    |   |   |
    │   │   ├── nets      
    │   │   ├── karras
    |   |   ├── ddpm
    |   |   └── trainers
    │   │
    │   └── metrics  <- Scripts to metrics for evaluating sampled models
    │       └── metrics.py   
    |
    | 
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── tests              <- Testing folder
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── saveddata
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |
    ├── savedmodels        <- Trained and serialized models, model predictions, or model summaries
    |
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>